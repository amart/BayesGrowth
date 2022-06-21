// from the CASAL manual, 2012
// The von Bertalanffy curve is parameterised by Linf, k, and t0; the Schnute curve (Schnute
// 1981) by y1 and y2, which are the mean sizes at reference ages τ1 and τ2, and a and b (when
// b=1, this reduces to the von Bertalanffy with k=a).

data {

  int<lower=1> n; //number of samples
  //data vectors
  vector<lower=0>[n] Age; //age data
  vector<lower=0>[n] Length; //length data

  //prior data
  vector[2] tau; // age at a small length, L1; age at a large length, L2
  vector[5] priors; //L2, L1, a, b, sigma
  vector<lower=0>[2] priors_se; //sd of L2, L1

}

parameters {
  //Schnute parameters
  real<lower=0> L1; //length at age tau1
  real<lower=0> L2; //length at age tau2
  real a; // growth coefficient
  real b; // growth coefficient

  //Likelihood parameters
  real<lower=0> sigma; //RSE
}

model {
  //storage
  vector[n] PredL; //predicted lengths

  //Schnute priors
  L2 ~ normal(priors[1], priors_se[1]);
  L1 ~ normal(priors[2], priors_se[2]);

  a ~ uniform(0, priors[3]);
  b ~ uniform(0, priors[4]);

  sigma ~ uniform(0, priors[5]);


  //Schnute likelihood
  tau_diff = tau[2] - tau[1];

  for(i in 1:n){
    // PredL[i] = Linf - (Linf - L0)*exp(-k*Age[i]);

    age_diff = Age[i] - tau[1];

    if (a != 0 && b != 0) {
        PredL[i] = ((L1^b) + ( ((L2^b) - (L1^b)) * (1 - exp(-a * age_diff)) / (1 - exp(-a * tau_diff)) ) )^(1/b);
    } else
    if (a != 0 && b == 0) {
        PredL[i] = L1 * exp( ln(L2 / L1) * (1 - exp(-a * age_diff)) / (1 - exp(-a * tau_diff)) );
    } else
    if (a == 0 && b != 0) {
        PredL[i] = ((L1^b) + ( ((L2^b) - (L1^b)) * age_diff / tau_diff ) )^(1/b);
    } else {
        PredL[i] = L1 * exp( ln(L2 / L1) * age_diff / tau_diff );
    }

    target += normal_lpdf(Length[i] | PredL[i], sigma);//likelihood

  }

}
// Individual loglikelihoods for loo


generated quantities {
    vector[n] log_lik;

    for (i in 1:n) {
      // log_lik[i] = normal_lpdf(Length[i]|(L2 - (L2 - L1)*exp(-k*Age[i])), sigma);

      if (a != 0 && b != 0) {
          log_lik[i] = normal_lpdf(Length[i] | ( ((L1^b) + ( ((L2^b) - (L1^b)) * (1 - exp(-a * age_diff)) / (1 - exp(-a * tau_diff)) ) )^(1/b) ), sigma);
      } else
      if (a != 0 && b == 0) {
          log_lik[i] = normal_lpdf(Length[i] | ( L1 * exp( ln(L2 / L1) * (1 - exp(-a * age_diff)) / (1 - exp(-a * tau_diff)) ) ), sigma);
      } else
      if (a == 0 && b != 0) {
          log_lik[i] = normal_lpdf(Length[i] | ( ((L1^b) + ( ((L2^b) - (L1^b)) * age_diff / tau_diff ) )^(1/b) ), sigma);
      } else {
          log_lik[i] = normal_lpdf(Length[i] | ( L1 * exp( ln(L2 / L1) * age_diff / tau_diff ) ), sigma);
      }

    }

}
