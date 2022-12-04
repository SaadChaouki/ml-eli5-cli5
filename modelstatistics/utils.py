import random
import numpy as np
import warnings


def sample_size_calculation(p: float, d: float = .05, z: float = 1.96, finite_population: bool = True,
                            population_size: int = None) -> int:
    """
        Sources:
            Daniel WW (1999). Biostatistics: A Foundation for Analysis in the Health Sciences. 7th edition.
                New York: John Wiley & Sons.
            Naing, L., T. Winn, and B. N. Rusli. 'Practical issues in calculating the sample size for prevalence studies
                Archives of orofacial Sciences 1 (2006): 9-14.

        Formula Without Finite Population
            n = Z^2P(1-P) / d^2
            where
                n: Sample Size
                Z^2: Z Statistics for a level of confidence
                P: Expected Prevalence
                d: Precision

        Formula With Finite Population
            n' = NZ^2P(1-P) / d^2(N-1) + Z^2P(1-P)
            where
                n': Sample size corrected for finite population
                N: Population size
                Z: Z Statistics for a level of confidence
                P: Expected Prevalence
                d: Precision

    !!: The finite population calculation should be used if n/N > .05. (Daniel, 1999).

    Setting d:
        Therefore, we recommend d as a half of P if P is below 0.1(10%)
        and if P is above 0.9 (90%), d can be {0.5(1-P)}. For example, if P is 0.04, investigators may use d=0.02,
        and if P is 0.98, we recommend d=0.01. (Naing, 2006)

    """

    # If the calculation is set to finite_population, the population size can't be None.
    if finite_population and not population_size:
        raise Exception('For finite population, set a population size.')

    # Without finite population
    if not finite_population:
        return int(np.round(((z ** 2) * p * (1 - p)) / (d ** 2), 0))

    # With finite population
    sample_size = np.round(
        (population_size * (z ** 2) * p * (1 - p)) / (((d ** 2) * (population_size - 1)) + ((z ** 2) * p * (1 - p))), 0)
    if sample_size / population_size < .05:
        warnings.warn('Computed sample size is less than 0.05 of population. Use infinite population calculation '
                      '(Daniel, 1999).', UserWarning)

    return int(sample_size)


def compute_confidence_intervals(x: np.array, z: float = 1.96) -> float:
    """
    Function to compute the confidence interval of the mean of a sample.

    Hazra, Avijit. "Using the confidence interval confidently." Journal of thoracic disease 9.10 (2017): 4125.
    Formula:
        CI = x̅ ± z × (std/√n)
        where
            CI: Confidence Interval
            x̅: Sample Mean
            z: Z Statistic for desired confidence interval
            std: Sample Standard Deviation
            n: Sample Size
    """

    return z * (x.std()/len(x)**.5)


if __name__ == '__main__':
    weighted_list = [1] * 2 + [0] * 98
    population = np.array([random.choice(weighted_list) for _ in range(1000)])
    computed_sample_size = sample_size_calculation(p=0.02, d=0.02, z=1.96, finite_population=True,
                                                   population_size=len(population))

    print(f'Population Size: {len(population)}')
    print(f'Computed Sample Size: {computed_sample_size}')

    # Selecting the sample
    sample = np.array(random.choices(population, k=computed_sample_size))
    confidence_interval = compute_confidence_intervals(sample)
    print('*'*50)
    print(f'True average: {population.mean()*100}')
    print(f'Sample Average: {sample.mean()*100} ± {confidence_interval*100}.')

