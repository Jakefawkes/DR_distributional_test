import doubleml as dml
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
from src.data import *
from zepid.causal.doublyrobust import TMLE
import statsmodels.api as sm
from scipy.stats import logistic, norm
from zepid.causal.doublyrobust.utils import tmle_unit_unbound
from zepid.calc import probability_to_odds

def double_ml_test(test_data):
    df = test_data.pd_df()
    X_cols = [col for col in df.columns if col.startswith("X")]
    dml_data = dml.DoubleMLData(df,y_col="Y",d_cols="T",x_cols = X_cols)
    learner = RandomForestRegressor(n_estimators = 500, max_features = 'sqrt', max_depth= 5)

    ml_l_bonus = clone(learner)

    ml_m_bonus = clone(learner)

    learner = LassoCV()

    ml_l_sim = clone(learner)

    ml_m_sim = clone(learner)
    obj_dml_plr_bonus = dml.DoubleMLPLR(dml_data, ml_l_bonus, ml_m_bonus)
    obj_dml_plr_bonus.fit()
    return obj_dml_plr_bonus.pval


class TMLE_pval(TMLE):
    def fit(self):
        """Calculate the effect measures from the predicted exposure probabilities and predicted outcome values using
               the TMLE procedure. Confidence intervals are calculated using influence curves.
               Note
               ----
               Exposure and outcome models must be specified prior to `fit()`
               Returns
               -------
               TMLE gains `risk_difference`, `risk_ratio`, and `odds_ratio` for binary outcomes and
               `average _treatment_effect` for continuous outcomes
               """
        if (self._fit_exposure_model is False) or (self._fit_outcome_model is False):
            raise ValueError('The exposure and outcome models must be specified before the psi estimate can '
                             'be generated')
        if self._miss_flag and not self._fit_missing_model:
            warnings.warn("No missing data model has been specified. All missing outcome data is assumed to be "
                          "missing completely at random. To relax this assumption to outcome data is missing at random"
                          "please use the `missing_model()` function", UserWarning)

        # Step 4) Calculating clever covariate (HAW)
        if self._miss_flag and self._fit_missing_model:
            self.g1W_total = self.g1W * self.m1W
            self.g0W_total = self.g0W * self.m0W
        else:
            self.g1W_total = self.g1W
            self.g0W_total = self.g0W
        H1W = self.df[self.exposure] / self.g1W_total
        H0W = -(1 - self.df[self.exposure]) / self.g0W_total
        HAW = H1W + H0W

        # Step 5) Estimating TMLE
        f = sm.families.family.Binomial()
        y = self.df[self.outcome]
        log = sm.GLM(y, np.column_stack((H1W, H0W)), offset=np.log(probability_to_odds(self.QAW)),
                     family=f, missing='drop').fit()
        self._epsilon = log.params
        Qstar1 = logistic.cdf(np.log(probability_to_odds(self.QA1W)) + self._epsilon[0] / self.g1W_total)
        Qstar0 = logistic.cdf(np.log(probability_to_odds(self.QA0W)) - self._epsilon[1] / self.g0W_total)
        Qstar = log.predict(np.column_stack((H1W, H0W)), offset=np.log(probability_to_odds(self.QAW)))

        # Step 6) Calculating Psi
        if self.alpha == 0.05:  # Without this, won't match R exactly. R relies on 1.96, while I use SciPy
            zalpha = 1.96
        else:
            zalpha = norm.ppf(1 - self.alpha / 2, loc=0, scale=1)

        # p-values are not implemented (doing my part to enforce CL over p-values)
        delta = np.where(self.df[self._missing_indicator] == 1, 1, 0)
        if self._continuous_outcome:
            # Calculating Average Treatment Effect
            Qstar = tmle_unit_unbound(Qstar, mini=self._continuous_min, maxi=self._continuous_max)
            Qstar1 = tmle_unit_unbound(Qstar1, mini=self._continuous_min, maxi=self._continuous_max)
            Qstar0 = tmle_unit_unbound(Qstar0, mini=self._continuous_min, maxi=self._continuous_max)

            self.average_treatment_effect = np.nanmean(Qstar1 - Qstar0)
            # Influence Curve for CL
            y_unbound = tmle_unit_unbound(self.df[self.outcome], mini=self._continuous_min, maxi=self._continuous_max)
            ic = np.where(delta == 1,
                          HAW * (y_unbound - Qstar) + (Qstar1 - Qstar0) - self.average_treatment_effect,
                          Qstar1 - Qstar0 - self.average_treatment_effect)
            seIC = np.sqrt(np.nanvar(ic, ddof=1) / self.df.shape[0])
            self.average_treatment_effect_se = seIC
            self.average_treatment_effect_ci = [self.average_treatment_effect - zalpha * seIC,
                                                self.average_treatment_effect + zalpha * seIC]
            self.pval = (1- norm.cdf(np.abs(self.average_treatment_effect)/self.average_treatment_effect_se))*2
                # norm.cdf()

        else:
            # Calculating Risk Difference
            self.risk_difference = np.nanmean(Qstar1 - Qstar0)
            # Influence Curve for CL
            ic = np.where(delta == 1,
                          HAW * (self.df[self.outcome] - Qstar) + (Qstar1 - Qstar0) - self.risk_difference,
                          (Qstar1 - Qstar0) - self.risk_difference)
            seIC = np.sqrt(np.nanvar(ic, ddof=1) / self.df.shape[0])
            self.risk_difference_se = seIC
            self.risk_difference_ci = [self.risk_difference - zalpha * seIC,
                                       self.risk_difference + zalpha * seIC]

            # Calculating Risk Ratio
            self.risk_ratio = np.nanmean(Qstar1) / np.nanmean(Qstar0)
            # Influence Curve for CL
            ic = np.where(delta == 1,
                          (1 / np.mean(Qstar1) * (H1W * (self.df[self.outcome] - Qstar) + Qstar1 - np.mean(Qstar1)) -
                           (1 / np.mean(Qstar0)) * (
                                       -1 * H0W * (self.df[self.outcome] - Qstar) + Qstar0 - np.mean(Qstar0))),
                          (Qstar1 - np.mean(Qstar1)) + Qstar0 - np.mean(Qstar0))

            seIC = np.sqrt(np.nanvar(ic, ddof=1) / self.df.shape[0])
            self.risk_ratio_se = seIC
            self.risk_ratio_ci = [np.exp(np.log(self.risk_ratio) - zalpha * seIC),
                                  np.exp(np.log(self.risk_ratio) + zalpha * seIC)]

            # Calculating Odds Ratio
            self.odds_ratio = (np.nanmean(Qstar1) / (1 - np.nanmean(Qstar1)
                                                     )) / (np.nanmean(Qstar0) / (1 - np.nanmean(Qstar0)))
            # Influence Curve for CL
            ic = np.where(delta == 1,
                          ((1 / (np.nanmean(Qstar1) * (1 - np.nanmean(Qstar1))) *
                            (H1W * (self.df[self.outcome] - Qstar) + Qstar1)) -
                           (1 / (np.nanmean(Qstar0) * (1 - np.nanmean(Qstar0))) *
                            (-1 * H0W * (self.df[self.outcome] - Qstar) + Qstar0))),

                          ((1 / (np.nanmean(Qstar1) * (1 - np.nanmean(Qstar1))) * Qstar1 -
                            (1 / (np.nanmean(Qstar0) * (1 - np.nanmean(Qstar0))) * Qstar0))))
            seIC = np.sqrt(np.nanvar(ic, ddof=1) / self.df.shape[0])
            self.odds_ratio_se = seIC
            self.odds_ratio_ci = [np.exp(np.log(self.odds_ratio) - zalpha * seIC),
                                  np.exp(np.log(self.odds_ratio) + zalpha * seIC)]

def tmle_test(test_data):
    df = test_data.pd_df()
    X_cols = [col for col in df.columns if col.startswith("X")]
    cov_string = " + ".join(X_cols)
    tmle = TMLE_pval(df, exposure='T', outcome='Y')
    tmle.exposure_model(cov_string, print_results=False)
    tmle.outcome_model('T + '+cov_string, print_results=False)
    tmle.fit()
    return tmle.pval


