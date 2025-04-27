#import os  # Module for interacting with the operating system
import time  # Module for time-related operations
import ujson  # Module for working with JSON data
from random import randint  # Module for generating random numbers
from typing import Dict, List, Any  # Type hinting imports

import requests  # Library for making HTTP requests
from bs4 import BeautifulSoup  # Library for parsing HTML data
from selenium import webdriver  # Library for browser automation
from selenium.common.exceptions import NoSuchElementException  # Exception for missing elements
from webdriver_manager.chrome import ChromeDriverManager  # Driver manager for Chrome (We are using Chromium based )
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By


def extract_abstract(pub_soup):
    try:
        sections = pub_soup.find_all('section', class_='page-section')
        for section in sections:
            subheader = section.find('h2', class_='subheader')
            if subheader and 'Abstract' in subheader.text:
                textblock = section.find('div', class_='textblock')
                if textblock:
                    return textblock.get_text(strip=True)
        return "No abstract available"
    except Exception as e:
        print(f"Error extracting abstract: {e}")
        return "Error fetching abstract"

def write_authors(list1, file_name):
     # Function to write authors' URLs to a file
    with open(file_name, 'w', encoding='utf-8') as f:
        for i in range(0, len(list1)):
            f.write(list1[i] + '\n')


def initCrawlerScraper(seed, max_profiles=500):
    # Initialize driver for Chrome
    webOpt = webdriver.ChromeOptions()
    webOpt.add_experimental_option('excludeSwitches', ['enable-logging'])
    webOpt.add_argument('--ignore-certificate-errors')
    webOpt.add_argument('--incognito')
    webOpt.headless = True

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=webOpt)
    driver.get(seed)  # Start with the original link

    Links = []  # Array with pureportal profiles URL
    pub_data = []  # To store publication information for each pureportal profile

    print("Crawler has begun...")
    while True:
        page = driver.page_source
        bs = BeautifulSoup(page, "lxml")  # Parse the page source

        # Extracting authors' profile URLs
        for link in bs.findAll('a', class_='link person'):
            url = str(link)[str(link).find('https://pureportal.coventry.ac.uk/en/persons/'):].split('"')[0]
            Links.append(url)

        # Click on Next button to visit next page
        try:
            next_button = driver.find_element(By.CSS_SELECTOR, ".nextLink")
            if next_button.is_enabled():
                driver.execute_script("arguments[0].click();", next_button)
            else:
                break
        except NoSuchElementException:
            break

        if len(Links) >= max_profiles:
            break

    print(f"Crawler has found {len(Links)} pureportal profiles")
    write_authors(Links, 'Authors_URL.txt')  # Write the authors' URLs to a file

    print(f"Scraping publication data for {len(Links)} pureportal profiles...")
    for link in Links:
        time.sleep(1)  # Small delay between requests
        driver.get(link)

        try:
            # Check if the profile has a 'Research Output' button
            research_outputs = driver.find_elements(By.CSS_SELECTOR, ".portal_link.btn-primary.btn-large")
            clicked = False
            for button in research_outputs:
                if "research output" in button.text.lower():
                    driver.execute_script("arguments[0].click();", button)
                    driver.get(driver.current_url)
                    clicked = True
                    break

            # Get author's name
            name_element = driver.find_element(By.CSS_SELECTOR, "div.header.person-details > h1")
            author_name = name_element.text

            # Request the current page for publications
            page_source = driver.page_source
            soup = BeautifulSoup(page_source, 'lxml')


            # Locate the publication list
            table = soup.find('ul', class_='list-results') if clicked else soup.find('div', class_='relation-list relation-list-publications')

            if table:
                for row in table.find_all('div', class_='result-container'):
                    data = {}
                    data['name'] = row.h3.a.text
                    data['pub_url'] = row.h3.a['href']
                    date_element = row.find("span", class_="date")
                    data['date'] = date_element.text if date_element else "No date"
                    data['cu_author'] = author_name

                    # Fetch abstract from publication page
                    try:
                        driver.get(data['pub_url'])
                        time.sleep(2)
                        pub_page = driver.page_source
                        pub_soup = BeautifulSoup(pub_page, 'lxml')
                        data['abstract'] = extract_abstract(pub_soup)
                    except Exception as e:
                        data['abstract'] = "Error fetching abstract"


                    pub_data.append(data)

        except Exception as e:
            print(f"Error processing profile {link}: {e}")
            continue

    print(f"Crawler has scraped data for {len(pub_data)} pureportal publications")
    driver.quit()

    # Writing all the scraped results to a file in JSON format
    with open('scraper_results.json', 'w', encoding='utf-8') as f:
        ujson.dump(pub_data, f,ensure_ascii=False, indent=4)



initCrawlerScraper('https://pureportal.coventry.ac.uk/en/organisations/coventry-university/persons/', max_profiles=50)



