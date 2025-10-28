import time
start_time = time.time()

import os
import sys
import glob
import numpy as np 
from pathlib import Path
from urllib import request
from ppxf.ppxf import ppxf
import ppxf.sps_util as lib
from astropy.io import fits
import ppxf.ppxf_util as util
from astropy.table import Table 
import matplotlib.pyplot as plt
from astropy import constants as const
plt.switch_backend('agg')

# This is the function that gives scores of proximity for each fitting.   
def score_closeness(values: list) -> list:
    scores = []
    n = len(values)

    for i in range(n):
        if n != 1:
            if i == 0:  # First value, only compare with the next value
                score = 1 / abs(values[i] - values[i+1])
            elif i == n-1:  # Last value, only compare with the previous value
                score = 1 / abs(values[i] - values[i-1])
            else:  # Compare with both previous and next values
                score = 1 / (abs(values[i] - values[i-1]) + abs(values[i] - values[i+1]))
        else:
            score = 0

        scores.append(score)
        normalized_scores = (np.array(scores) - np.min(scores)) / (np.max(scores))

    return normalized_scores

ppxf_dir = Path(util.__file__).parent
sample = 'bigw2m'

# Get list of spectra to process
sources_list = glob.glob(f'/home/ether/ether-ppxf/out/main_query/spectra/{sample}/*.fits')

for source in sources_list[-2:-1]:
    # Empty lists to store results
    original_sigma = []
    ppxf_first_sigma = []
    ppxf_conv_sigma = []
    runs_used = []
    good_range = []
    plot_data = {}
    
    name = source.split('_')[-1][:-5]
    print(f'\nSOURCE: {name}\n')
    
    # Extract data and header from FITS file
    hdul = fits.open(source)
    data_org = hdul[1].data
    header = hdul[1].header

    # Define variables
    mbh = header['MBHWISE'] # Log-scale WISE-based black hole mass estimate
    redshift_0 = header['Z'] # SDSS redshift estimate
    sigma_0 = 0

    lam_gal_org = 10**data_org['loglam'] # Observed-frame wavelength in Angstroms in every pixel (log sampled)
    
    # Masking observed-frame wavelength (in Angstroms) range if needed
    lam_lims = {
        'all': [min(lam_gal_org), max(lam_gal_org)], # Wavelength limits in Angstroms
        '5kto6k': np.array([5000, 6000])*(1 + redshift_0),
        'mgi': np.array([5000, 5300])*(1 + redshift_0), # MgI region
        'nai': np.array([8050, 8350])*(1 + redshift_0), # NaI region
        'cat': np.array([8500, 8800])*(1 + redshift_0), # CaT region
        }

    for key, value in lam_lims.items():
        data = data_org[(lam_gal_org > value[0]) & (lam_gal_org < value[1])]
        lam_gal = lam_gal_org[(lam_gal_org > value[0]) & (lam_gal_org < value[1])]

        galaxy = data['flux']/np.median(data['flux']) # Normalize spectrum to median value to avoid numerical issues
        lam_gal *= np.median(util.vac_to_air(lam_gal)/lam_gal) # Converts the wavelength array from vacuum to air scale by applying a single median correction factor

        # Calculate velocity scale of the spectra in km/s per pixel
        c = const.c.to('km/s').value # Speed of light in km/s
        velscale = c*np.diff(np.log(lam_gal)).mean() # Velocity scale in km/s

        # Calculate the spectral resolution FWHM of the galaxy spectrum in Angstroms
        dlam_gal = np.gradient(lam_gal) # Size of every pixel in Angstroms
        wdisp = data['wdisp'] # Instrumental dispersion of every pixel, in pixels units
        fwhm_gal = 2.355*wdisp*dlam_gal # Resolution FWHM of every pixel, in Angstroms

        # Adjust wavelength and resolution to rest-frame
        lam_gal = lam_gal/(1 + redshift_0) # Compute approximate rest-frame wavelength
        fwhm_gal = fwhm_gal/(1 + redshift_0) # Adjust resolution in Angstroms

        # Calculate noise (Standard deviation) spectrum from inverse variance
        noise = 1/np.sqrt(data['ivar'])
        noise[noise == np.inf] = max(noise[noise != np.inf], default = 0) # Replace infinite noise values with the maximum finite noise value

        # Load Stellar Population Synthesis (SPS) model templates
        # Choose one of the following SPS models by uncommenting the desired line
        sps_name = 'emiles'
        # sps_name = 'fsps'
        # sps_name = 'galaxev'
        # sps_name = 'xsl'

        # Download the SPS model file if it doesn't exist locally
        basename = 'spectra_{}_9.0.npz'.format(sps_name)
        filename = ppxf_dir / 'sps_models' / basename
        if not filename.is_file():
            url = "https://raw.githubusercontent.com/micappe/ppxf_data/main/" + basename
            request.urlretrieve(url, filename)

        lam_range_temp = [1000, 15000] # Wavelength range of the SPS templates in Angstroms
        fwhm_gal = {"lam": lam_gal, "fwhm": fwhm_gal} # Dictionary with wavelength and FWHM arrays for the galaxy spectrum in Angstroms

        sps = lib.sps_lib(filename, velscale, fwhm_gal, lam_range = lam_range_temp) # Load SPS templates with appropriate velocity scale and resolution

        goodpixels = util.determine_goodpixels(np.log(lam_gal), lam_range_temp) # Identify good pixels within the specified wavelength range

        # Define initial guesses and bounds for pPXF fitting
        sigma_init = 100 # Initial guess for velocity dispersion in km/s
        vel_init = 0 # Initial guess for velocity in km/s (i.e., spectrum is de-redshifted)
        start = [vel_init, sigma_init] # (km/s), starting guess for [V, sigma]

        bound_vel = [-2000,2000]
        bound_sigma = [10,1200]
        bound_h3, bound_h4 = [-0.3, 0.3], [-0.3, 0.3]
        bounds_stars = [bound_vel, bound_sigma, bound_h3, bound_h4]

        # Lists to store pPXF fitting results
        predt_sigmas = []
        predt_errors = []
        pps = []
        chi2 = []
        dof = []
        starting_wavelengths = []

        # Here is where the code drop pixels to change the spectra and save the calculated sigma at the end
        if key == 'all':
            TEST_GP_ITERATIVE = np.arange(0, int(len(goodpixels)/2), 100)
        else:
            TEST_GP_ITERATIVE = [0] # Only fit the full goodpixels range for specific regions
        
        for GP in TEST_GP_ITERATIVE:
            try:
                pp = ppxf(sps.templates, galaxy, noise, velscale, start, bounds = bounds_stars, clean = True,
                        goodpixels = goodpixels[GP:], plot = False, moments = 4, trig = 1,
                        degree = 10, lam = lam_gal, lam_temp = sps.lam_temp)
                errors = pp.error*np.sqrt(pp.chi2)  # Assume the fit is good chi2/DOF=1

                # Extract fitted sigma and its error
                sigma_temp = pp.sol[1]
                error_temp = errors[1]
                
                # Store the starting wavelength corresponding to the first good pixel used in the fit
                starting_idx = goodpixels[GP]
                starting_wavelength = lam_gal[starting_idx]

                # Append results to lists
                starting_wavelengths.append(starting_wavelength)
                pps.append(pp)
                predt_sigmas.append(sigma_temp)
                predt_errors.append(error_temp)
                chi2.append(pp.chi2)
                dof.append(pp.dof)
            except:
                print('Error for fitting in {} pixels dropped'.format(GP))
        
        if pps == []:
            if len(lam_lims) == 1:
                print('\nNo successful pPXF fits for wavelength region: {}\n'.format(key))
                sys.exit()
            else:
                print('\nNo successful pPXF fits for wavelength region: {}\n'.format(key))
                continue # Skip to next wavelength range if no successful fits
        else:
            good_range.append(key)

        ppxf_first_sigma.append(predt_sigmas[0])
        original_sigma.append(sigma_0)
        chi2 = np.array(chi2)
        dof = np.array(dof)
        starting_wavelengths = np.array(starting_wavelengths)

        print('\nOriginal sigma: {:.2f} and pPXF result fitting to complete spectra: {:.2f}\n'.format(sigma_0, predt_sigmas[0]))

        # Clean list of sigma values by removing nonsense values
        sensitivity = 5 # km/s threshold for convergence
        if len(starting_wavelengths) > 1:
            converg_sigmas = predt_sigmas.copy()
            nonsense_bool = (np.array(converg_sigmas) > 1000) | (np.array(converg_sigmas) == sigma_init)
            sense_idx_list = np.where(~nonsense_bool)
            pps = np.array(pps)[sense_idx_list]
            converg_sigmas = np.array(converg_sigmas)[sense_idx_list]

            # If all values are nonsense, set converg_sigmas to zeros to avoid errors
            if len(converg_sigmas) == 0: 
                converg_sigmas = np.zeros_like(predt_sigmas)

            # Iteratively remove outliers based on weighted deviation until convergence
            converg_test = converg_sigmas - np.mean(converg_sigmas)
            weighted_deviation = (1 - score_closeness(converg_sigmas)) * abs(converg_test)

            accepted_bool = ~nonsense_bool  # Initially, values not marked as nonsense are accepted

            while any(weighted_deviation >= sensitivity):
                if len(converg_sigmas) <= 4:
                    break
                id_remove = np.argmax(weighted_deviation)  # Index to remove
                idx_in_predt = sense_idx_list[0][id_remove]  # Map index back to original predt_sigmas
                converg_sigmas = np.delete(converg_sigmas, id_remove)
                pps = np.delete(pps, id_remove)

                converg_test = converg_sigmas - np.mean(converg_sigmas)
                weighted_deviation = (1 - score_closeness(converg_sigmas)) * abs(converg_test)
                accepted_bool[idx_in_predt] = False  # Mark this index as not accepted

            # Store number of runs used for convergence
            runs_used.append(sum(accepted_bool))

            # Final converged sigma is the mean of the remaining values
            converg_sigma_final = np.mean(converg_sigmas)
            ppxf_conv_sigma.append(converg_sigma_final)
        
        elif len (starting_wavelengths) == 1:
            converg_sigma_final = predt_sigmas[0]
            ppxf_conv_sigma.append(converg_sigma_final)
            nonsense_bool = np.array([False])
            accepted_bool = np.array([True])
            runs_used.append(1)

        print('Converged sigma: {:.2f} using {} run(s) for a dist of {} m/s'.format(converg_sigma_final, sum(accepted_bool), sensitivity))

        # Store results per region for later combined plotting
        plot_data[key] = {
            'starting_wavelengths': starting_wavelengths.copy(),
            'predt_sigmas': np.array(predt_sigmas).copy(),
            'predt_errors': np.array(predt_errors).copy(),
            'chi2': np.array(chi2).copy(),
            'dof': np.array(dof).copy(),
            'nonsense_bool': nonsense_bool.copy(),
            'accepted_bool': accepted_bool.copy(),
            'converg_sigma_final': converg_sigma_final
        }

        if name not in os.listdir(f'/home/ether/ether-ppxf/out/fitting_pipeline/{sample}'):
            os.mkdir(f'/home/ether/ether-ppxf/out/fitting_pipeline/{sample}/{name}')

        # === Individual plot for the current wavelength range ===
        plt.figure(figsize = (10, 5), dpi = 150)
        pps[0].plot()
        for line in plt.gca().get_lines():
            line.set_linewidth(0.5) # Thin lines for better visualization
        plt.title(f"{name} — Wavelength region: {key.upper()} — Predicted WISE $M_{{BH}}$: {mbh:.2f} [log $M_\\odot$]")
        # Add a dummy line for the legend showing converged sigma
        plt.plot([], [], linestyle='-', color='red',
         label=fr'pPXF$_\rightarrow$ $\sigma$ = {converg_sigma_final:.2f} km/s')
        plt.legend(loc = 'best')
        plt.grid(alpha = 0.3)
        plt.savefig(f'/home/ether/ether-ppxf/out/fitting_pipeline/{sample}/{name}/sigma-fitting-{name}-{key}.png', transparent = False)

    # === Combined plot for all wavelength ranges ===
    if good_range != []:
        plt.figure(figsize=(10, 6), dpi=150)

        # Define colors and markers for each region
        colors = {'cat': 'royalblue', 'nai': 'forestgreen', 'mgi': 'darkorange', 'all': 'gray'}
        markers = {'cat': 'o', 'nai': 's', 'mgi': '^', 'all': 'x'}

        for key, pdata in plot_data.items():
            # Scatter points with different marker per key
            sc = plt.scatter(
                x=pdata['starting_wavelengths'][pdata['accepted_bool']],
                y=pdata['predt_sigmas'][pdata['accepted_bool']],
                c=(pdata['chi2'] * pdata['dof'])[pdata['accepted_bool']],
                cmap='viridis',
                edgecolor='k',
                s=70,
                marker=markers.get(key, 'o'),
                alpha=0.8,
                label=f"{key.upper()} region"
            )

            # Error bars
            plt.errorbar(
                x=pdata['starting_wavelengths'][pdata['accepted_bool']],
                y=pdata['predt_sigmas'][pdata['accepted_bool']],
                yerr=pdata['predt_errors'][pdata['accepted_bool']],
                fmt='none',
                ecolor='k',
                capsize=2,
                alpha=0.5
            )

            # Horizontal line for converged sigma across full x-axis
            plt.axhline(
                y=pdata['converg_sigma_final'],
                color=colors.get(key, 'black'),
                linestyle='--',
                linewidth=1,
                label=fr"{key.upper()} $\sigma_{{conv}}$={pdata['converg_sigma_final']:.2f}"
            )
            if key == 'all':
                plt.ylim(min(pdata['predt_sigmas'][pdata['accepted_bool']]) - 10, max(pdata['predt_sigmas'][pdata['accepted_bool']]) + 10)

        plt.xlabel(r'$\lambda_{\rm{rest}} (\AA)$')
        plt.ylabel(r'Predicted $\sigma_{\rm pPXF}$ (km/s)')
        plt.title(f"{name} — Predicted WISE $M_{{BH}}$: {mbh:.2f} [log $M_\\odot$]")
        plt.legend(loc = 'best')
        plt.grid(alpha=0.3)
        plt.colorbar(sc, label=r'$\chi^2$')
        plt.tight_layout()
        plt.savefig(f'/home/ether/ether-ppxf/out/fitting_pipeline/{sample}/{name}/sigma-guesses-pixels-{name}-combined.png', transparent=False)
        plt.close()
        
        # Table of the run is saved with all the necessary data
        data = Table([np.repeat(name, len(ppxf_conv_sigma)), np.repeat(mbh, len(ppxf_conv_sigma)), ppxf_first_sigma, ppxf_conv_sigma, runs_used, good_range], names = ['NAME', f'MBH-{sample}', 'sigma-first', 'sigma-conv', 'runs-used', 'wavelength-region'])
        data.write(f'/home/ether/ether-ppxf/out/fitting_pipeline/{sample}/{name}/results-{name}.fits', overwrite = True)

print(f'\nThe whole script execution took: {int(time.time() - start_time)} s\n')