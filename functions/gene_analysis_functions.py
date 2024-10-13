'''
functions that manipulate the web browser for the purposes of data analysis
'''
from selenium import webdriver

from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service

from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait, Select

from selenium.webdriver.common.by import By

from bs4 import BeautifulSoup

import os
import requests

import matplotlib.pyplot as plt
import numpy as np

import string
import time

from liftover import get_lifter



def get_cistrome(probe_data, fig_w=2800, fig_h=3000, 
                 check_coef=True, top_10k=False):

    '''
    generates cistrome plots using the inputted data

        param probe_data: dataset to be used, df, must contain probe position data, df
        param fig_w: figure width for cistrome subplots, int
        param fig_h: figure height for cistrome subplots, int
        check_coef: if the code should check for columns with '_coef' to determine how to rank probes,
            ranking with the highest absolute coef being moved to the top, etc., bool.
        top_10k: if the code should utilize the top 10k probes from the dataset, or use the default 1k, bool

        return: none
    '''

    # establish settings
    options = Options()

    # chrome options
    options = Options()
    options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
    options.add_argument("--headless")  
    options.add_argument("--incognito")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)

    # establish webdriver
    #driver = webdriver.Chrome(service=Service(), options=options)

    # avoid detection by web security
    #driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")

    ############

    # iterate through data and produce graphs
    images = []
    labels = []
    trait_factors = {}
    
    n_traits = 0
    for column in probe_data.columns:
        if (('_coef' in str(column)) or (not check_coef)):

            # prep data for analysis
            if check_coef:
                bed_data = probe_data[[column,'chr_mm10', 'pos_mm10']]
            else: # if check_coef is false, then column is an unwanted duplicate
                bed_data = probe_data[['chr_mm10', 'pos_mm10']]
            
            bed_data = bed_data[bed_data[column].notna()] # keep defined values
            if check_coef:
                bed_data = bed_data.sort_values(by=column, key=abs) # sort for desired order

            if top_10k: bed_data = bed_data.head(10000) # top 10k
            else: bed_data = bed_data.head(1000) # top 1k

            # put this into actual bed format
            bed_data['end_mm10'] = bed_data['pos_mm10']+2
            if check_coef: bed_data = bed_data.drop(columns=[column])
            if bed_data.empty:
                print(f'No valid probes, skipping trait {column}')
                continue

            # create target file
            bed_data.to_csv('temp.txt', sep='\t', index=False, header=False)

            # re-establish driver
            driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()),
                            options=options)

            # direct driver to Cistrome
            driver.get('http://dbtoolkit.cistrome.org')

            # change assembly to mm10
            select = Select(driver.find_element(By.XPATH, '/html/body/div[4]/div/div/div/div[2]/div/form/div[1]/div[2]/div[1]/div/select'))
            select.select_by_visible_text('Mouse mm10')

            # if you want the top 10k probes to be used, select the top 10k option
            if top_10k:
                select = Select(driver.find_element(By.XPATH, '/html/body/div[4]/div/div/div/div[2]/div/form/div[1]/div[4]/div[1]/div/select'))
                select.select_by_visible_text('Top 10k peaks according to peak enrichment')
            else:
                pass

            # select 'choose file' button
            select_file = WebDriverWait(driver, 20).until(EC.element_to_be_clickable((By.ID, 'peak')))

            # upload probe set
            select_file.send_keys(str(os.getcwd()) + '/temp.txt')

            # press submit button
            submit_button = driver.find_element(By.XPATH, '/html/body/div[4]/div/div/div/div[2]/div/form/div[2]/div[2]/input')
            submit_button.click()

            # get the biosample associated factors
            bio_factors = {}

            try: soup = BeautifulSoup(driver.page_source, 'lxml')
            except: soup = BeautifulSoup(driver.page_source, 'html.parser')

            i = 1
            while i < 10:
                tables = soup.find_all('table', class_='table table-bordered table-hover') # main table
                try:
                    all_associations = tables[0].find_all('tr')[1:] # all associations on the first page, skipping the header
                except IndexError:
                    bio_factors['None'] = None
                    break
                for sample in all_associations: # for row in table
                    info = sample.select('td')
                    if info[5].get_text(strip=True) == 0: # if a GIGGLE score of 0
                        break
                    if info[3].get_text(strip=True) in bio_factors:
                        bio_factors[info[3].get_text(strip=True)].append(info[4].get_text(strip=True)) # add factor to dictionary
                    else:
                        bio_factors[info[3].get_text(strip=True)] = [info[4].get_text(strip=True)]
                
                # go to the next page
                i+=1
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);") # scroll to the bottom of the page
                next_button = driver.find_element(By.LINK_TEXT, str(i))
                next_button.click()

            if check_coef:
                trait_factors[column[:-5]] = bio_factors
            else:
                trait_factors[column] = bio_factors

            driver.execute_script("window.scrollTo(0, 0);") # scroll to the top of the page

            # select figures
            screenshot(driver)
            show_plot = driver.find_element(By.LINK_TEXT, 'Result in figure')
            show_plot.click()

            # select iframe
            iframe = driver.find_element(By.XPATH, '/html/body/div[2]/div/div/div/div[2]/div[3]/div/div[2]/iframe')
            driver.switch_to.frame(iframe)

            # select static plot
            static = driver.find_element(By.ID, 'staticli')
            static.click()

            # make plot larger
            plot_full = driver.find_element(By.LINK_TEXT, 'Download')
            plot_full.click()
            driver.switch_to.window(driver.window_handles[1]) #focus driver on newly opened tab

            # save plot
            png_element = driver.find_element(By.TAG_NAME, 'img')
            png_url = png_element.get_attribute('src')
            response = requests.get(png_url)

            # save image to working directory
            question_mark = False
            if check_coef:
                file_name = f'{column[:-5]}_temp_img.png'
            else:
                file_name = f'{column}_temp_img.png'
            if '?' in file_name:
                file_name = file_name.replace('?', 'xYZabC')
                question_mark = True
            try:
                with open(file_name, 'wb') as file:
                    file.write(response.content)
            except FileNotFoundError:
                if '/' in file_name:
                    file_name = file_name.replace('/', '_')
                    with open(file_name, 'wb') as file:
                        file.write(response.content)

                else:
                    print('Error: illegal character in trait name, modify exception accordingly')
                    pass
            path = os.path.join(os.getcwd(), file_name)
        
            images.append(path)

            if question_mark: file_name = file_name.replace('xYZabC', '?') # if there is a question mark
            labels.append(file_name[:-13])

            n_traits+=1

            if check_coef == False:
                break
        else:
            pass


    # establish subplots 
    px = 1/plt.rcParams['figure.dpi']
    if n_traits % 2 == 0:
        fig, axs = plt.subplots(n_traits, 2, figsize=(fig_w*px, fig_h*px))
    else:
        fig, axs = plt.subplots((n_traits//2)+1, 2, figsize=(fig_w*px, fig_h*px))

    # display images on subplots
    for i, ax in enumerate(axs.flat):
        try:
            plot = plt.imread(images[i])
            ax.imshow(plot, extent=[0, 1, 0, 1], aspect='auto')
            ax.axis('off')
            if check_coef: ax.set_title(labels[i], size=20)
        except IndexError:
            fig.delaxes(ax)
            pass

        # add subplot indicator text
        ax.text(0.1, 1.1, string.ascii_uppercase[i], transform=ax.transAxes, 
            size=25, weight='bold')

    # remove temp file
    os.remove('temp.txt')

    # remove images
    for image in images:
        os.remove(image)

    driver.quit()
    return trait_factors

def get_pos(probe_data, ref_data, drop_undef=True):
    '''
    gets genetic information from illumina reference table, as provided, and prints out BED information

        param probe_data: data from which assessed probe names are derived, requires index to be probe names, df
        param ref_data: data indicating the position of probes, df
        param drop_undef: if True, drop undefined probes

        return: probe_data with chromosome and position added as columns, df
    '''

    chr_mm39 = []
    pos_mm39 = []

    chr_mm10 = []
    pos_mm10 = []

    converter = get_lifter('mm39', 'mm10', one_based=True)
    n_probes = probe_data.shape[0]

    n = 0
    while n < n_probes:
        bed_data = ref_data.loc[probe_data.index[n]]
        chr = bed_data.loc['chr']
        pos = bed_data.loc['start']

        if (chr == 'nan') or (pos == 'nan'):
            chr_mm10.append(np.nan)
            pos_mm10.append(np.nan)

        else:
            chr_mm39.append(chr)
            pos_mm39.append(int(pos))

            try:
                chr_mm10.append(converter[chr][pos][0][0])
                pos_mm10.append(int(converter[chr][pos][0][1]))
            except (IndexError, KeyError, TypeError): # IndexError: No conversion, KeyError/TypeError: Initial file gives nan
                chr_mm10.append(np.nan)
                pos_mm10.append(np.nan)

        n+=1

    probe_data['chr_mm39'] = chr_mm39
    probe_data['pos_mm39'] = pos_mm39
    probe_data['end_mm39'] = probe_data['pos_mm39']+2

    probe_data['chr_mm10'] = chr_mm10
    probe_data['pos_mm10'] = pos_mm10
    probe_data['end_mm10'] = probe_data['pos_mm10']+2

    if drop_undef:
        probe_data = probe_data.dropna()
    else:
        pass

    return probe_data

def screenshot(driver):
    '''
    generates a screenshot corresponding to the current state of the driver
        param driver: the driver to assess

        return: none
    '''
    time.sleep(3)
    driver.set_window_size(1024, 600)
    driver.save_screenshot('screenshot.png')
    return

def insig_nan(data):
    '''
    replaces insiginficant p values (> 0.01) with nan
        param data: data to be modified

        return: modified data
    '''
    for column in data.columns: 
        if '_pval' in column:
            data[column] = np.where(data[column] > 0.01, np.nan, data[column])

    return(data)