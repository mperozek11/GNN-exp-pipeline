from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import time
import pandas as pd
import multiprocessing
import pandas as pd


class Listing:
        def __init__(self, id: str, 
                    title: str, price: float, 
                    sellerName: str, url: str):
            self.id = id
            self.title = title
            self.price = price
            self.sellerName = sellerName
            self.url = url
            self.tier = None
            self.BE = None
            self.server = None

        def __str__(self):
            return self.title
        
        # getters and setters
        def getTitle(self):
            return self.title
        def setTitle(self, title):
            self.title = title
        def getPrice(self):
            return self.price
        def setPrice(self, price):
            self.price = price
        def getSellerName(self):
            return self.sellerName
        def setSellerName(self, seller):
            self.sellerName = seller
        def getURL(self):
            return self.url
        def setURL(self, link):
            self.url = link
        def getId(self):
            return self.id
        def setId(self, id):
            self.id = id

class G2GListing(Listing):
    def __init__(self, id: str, 
                    title: str, price: float, 
                    sellerName: str, url: str,
                    tier: str, server: str):
        super().__init__(id, title, price, sellerName, url)
        self.tier = tier
        self.server = server

    def __str__(self):
        return f'id: {self.id},' + f'\n title: {self.title},' + f'\n price: {self.price},' + f'\n seller: {self.sellerName},\n url: {self.url},\n tier: {self.tier},\n server: {self.server}'

    # getters and setters
    def getServer(self):
        return self.server
    def setServer(self, server):
        self.server = server
        
    def to_pd_row(self):
        return pd.DataFrame(
            data=[[
            self.id,
            self.title,
            self.price,
            self.sellerName,
            self.url,
            self.tier,
            self.server,]], 
            columns=[
                'id', 'title', 'price',
                'seller_name', 'url', 'tier', 
                'server'])


class PAListing(Listing):
    def __init__(self, id, title, price, seller_name, url, total_orders, member_since, stars, reviews, delivery):
        super().__init__(id, title, price, seller_name, url)
        self.total_orders = total_orders
        self.member_since = member_since
        self.stars = stars
        self.reviews = reviews
        self.delivery = delivery

    def __str__(self):
        return str({
            'id': self.id,
            'title': self.title,
            'price': self.price,
            'seller_name': self.seller_name,
            'url': self.url,
            'tier': self.tier,
            'BE': self.BE,
            'server': self.server,
            'total_orders': self.total_orders,
            'member_since': self.member_since,
            'stars': self.stars,
            'reviews': self.reviews,
            'delivery': self.delivery
        })

    def to_pd_row(self):
        return pd.DataFrame(
            data=[[
            self.id,
            self.title,
            self.price,
            self.seller_name,
            self.url,
            self.tier,
            self.BE,
            self.server,
            self.total_orders,
            self.member_since,
            self.stars,
            self.reviews,
            self.delivery]], 
            columns=[
                'id', 'title', 'price',
                'seller_name', 'url', 'tier', 
                'BE', 'server', 'total_orders',
                'member_since', 'stars', 'reviews', 'delivery'])
    

PA_BASE_URL = 'https://www.playerauctions.com/lol-account/'

def scrape_full_PA(url, pages=5):
    options = webdriver.FirefoxOptions()
    # options.add_argument('-headless')
    browser = webdriver.Firefox(options = options)
    browser.get(url)
    close_cookies = browser.find_element(By.XPATH, '//a[@id="close-cross"]')
    close_cookies.click()

    all_listings = []
    i = 0
    all_listings += scrape_PA_page(browser)
    next_page = browser.find_element(By.XPATH, '/html/body/main/div/div[1]/div[2]/div[6]/nav/ul/li[7]/a')
    href_data = next_page.get_attribute('href')
    while href_data:
        try:
            next_page.click()
        except:
            print('exception')
            break
        
        all_listings += scrape_PA_page(browser)
        next_block = WebDriverWait(browser, 15).until(EC.presence_of_element_located((By.XPATH,'//*[@id="pagination"]')))
        if len(next_block.find_elements(By.CLASS_NAME, 'disabled')) > 0:
            print('end handling correct')
            break
        next_page = WebDriverWait(browser, 15).until(EC.presence_of_element_located((By.XPATH, '/html/body/main/div/div[1]/div[2]/div[6]/nav/ul/li[7]/a')))
        # next_page = browser.find_element(By.XPATH, '/html/body/main/div/div[1]/div[2]/div[6]/nav/ul/li[7]/a')
        href_data = next_page.get_attribute('href')
        i += 1
        if i % 5 == 0:
            print(f'scraping page{i}')

    return all_listings
    

def scrape_PA_page(browser, cookies=False):
    
        
    listings = browser.find_elements(By.CLASS_NAME, 'offer-item')
    
    offers = []
    for i in range(len(listings)):
        title = listings[i].find_element(By.CLASS_NAME, 'offer-title').text
        seller = listings[i].find_element(By.CLASS_NAME, 'offer-seller-name').text
        price = listings[i].find_element(By.CLASS_NAME, 'offer-price-tag').text
        delivery = listings[i].find_element(By.CLASS_NAME, 'OLP-delivery-text').text

        link_ext = listings[i].find_element(By.CLASS_NAME, 'txt-hot').get_attribute('href')
        list_id = link_ext.split('/')[-2]

        seller_info = listings[i].find_elements(By.CLASS_NAME, 'text-left')[1].text.split('\n')
        total_orders = seller_info[0].split()[-1]
        member_since = seller_info[1].split()[-1]
        stars = seller_info[2]
        reviews = seller_info[3][1:-1]

        offer = PAListing(
            id=list_id,
            title=title,
            price=price,
            seller_name=seller,
            url=link_ext,
            total_orders=total_orders,
            member_since=member_since,
            stars=stars,
            reviews=reviews,
            delivery=delivery
        )
        offers.append(offer)
        
    return offers

def scrape_full_PA_v2():
    options = webdriver.FirefoxOptions()
    options.add_argument('-headless')
    browser = webdriver.Firefox(options = options)
    browser.get(PA_BASE_URL)
    close_cookies = browser.find_element(By.XPATH, '//a[@id="close-cross"]')
    close_cookies.click()

    all_listings = []

    all_listings += scrape_PA_page_v2(browser)
    next_page = browser.find_element(By.XPATH, '/html/body/main/div/div[1]/div[2]/div[6]/nav/ul/li[7]/a')
    href_data = next_page.get_attribute('href')
    while href_data:
        try:
            next_page.click()
        except:
            print('exception in page end')
            break
        
        all_listings += scrape_PA_page_v2(browser)
        next_block = WebDriverWait(browser, 15).until(EC.presence_of_element_located((By.XPATH,'//*[@id="pagination"]')))
        if len(next_block.find_elements(By.CLASS_NAME, 'disabled')) > 0:
            print('end handling correct')
            break
        next_page = WebDriverWait(browser, 15).until(EC.presence_of_element_located((By.XPATH, '/html/body/main/div/div[1]/div[2]/div[6]/nav/ul/li[7]/a')))
        # next_page = browser.find_element(By.XPATH, '/html/body/main/div/div[1]/div[2]/div[6]/nav/ul/li[7]/a')
        href_data = next_page.get_attribute('href')

def scrape_PA_page_v2(browser):
    chunks = 4
    page_listings = []
    load_times = []
    listings = browser.find_elements(By.CLASS_NAME, 'offer-item')
    list_links = [l.find_element(By.CLASS_NAME, 'txt-hot').get_attribute('href') for l in listings]
    print(list_links)
    pool = multiprocessing.Pool()
    st = time.time()
    results = pool.map(scrape_single_listing, list_links)
    pool.close()
    pool.join()

    print(time.time()-st)
    # for l in listings:
    #     st = time.time()
    #     link = l.find_element(By.CLASS_NAME, 'txt-hot')
    #     link = link.get_attribute('href')
    #     # print(link)
    #     options = webdriver.FirefoxOptions()
    #     options.add_argument('-headless')
    #     b = webdriver.Firefox(options = options)
    #     b.get(link)
    #     offer_ends = b.find_element(By.CLASS_NAME, 'pa-xs-12').text
    #     print(offer_ends)
    #     # print(offer_ends)
    #     b.quit()
    #     load_times.append(time.time()-st)
    
    print(load_times)
        
def scrape_single_listing(url):
    options = webdriver.FirefoxOptions()
    options.add_argument('-headless')
    b = webdriver.Firefox(options = options)
    b.get(url)
    offer_ends = b.find_element(By.CLASS_NAME, 'pa-xs-12').text
    print(offer_ends)
    # print(offer_ends)
    b.quit()
    return offer_ends

def calculate_rank_from_pa_listing_text(df):
    unranked = ['unranked', '30']
    iron = ['iron']
    bronze = ['bronze']
    silver = ['silver']
    gold = ['gold']
    plat = ['platinum', 'plat']
    diamond = ['diamond']
    master = ['master']
    grandmaster = ['grandmaster', 'gm']
    challenger = ['challenger']

    all_wordlists = [unranked, iron, bronze, silver, gold, plat, diamond, master, grandmaster, challenger]

    rank_id_dict = {}

    for wl in all_wordlists:
        rank_id_dict.update({k:wl[0] for k in wl})

    no_rank_word = []
    multiple_rank_words = []
    rank = []
    for i, title in enumerate(df['title']):
        title = title.lower()
        matches = 0
        found = False
        for w in rank_id_dict.keys():
            if w in title:
                matches += 1
                if not found:
                    rank.append(rank_id_dict[w])
                    found = True
        if matches == 0:
            rank.append('unknown')
            no_rank_word.append(i)
        if matches > 1:
            multiple_rank_words.append(i)
            
    df['tier'] = rank
    return df

def calculate_server_from_pa_listing_text(df):
    br = ['brazil', 'br', 'br1']
    eune = ['europe nordic and east', 'eune', 'eun1']
    euw = ['europe west', 'euw', 'euw1']
    lan = ['latin america north', 'lan', 'la1']
    las = ['latin america south', 'las', 'la2']
    na = ['north america', 'na', 'na1']
    oce = ['oceania', 'oce', 'oc1']
    ru = ['russia', 'ru', 'ru1']
    tr = ['turkey', 'tr', 'tr1']
    jp = ['japan', 'jp', 'jp1']

    all_wordlists = [eune, euw, lan, las, na, oce, ru, tr, jp, br]

    server_id_dict = {}

    for wl in all_wordlists:
        server_id_dict.update({k:wl[1] for k in wl})

    no_server_word = []
    multiple_server_words = []
    server = []
    for i, title in enumerate(df['title']):
        title = title.lower()
        matches = 0
        found = False
        for w in server_id_dict.keys():
            if w in title:
                if w == 'br' and 'bronze' in title: # bronze edge case this will not pick up accounts which are bronze rank, brazil server if only server data is 'br'
                    pass
                else:
                    matches += 1
                    if not found:
                        server.append(server_id_dict[w])
                        found = True
        if matches == 0:
            server.append('no server')
            no_server_word.append(i)
        if matches > 1:
            multiple_server_words.append(i)
    
    df['server'] = server
    return df

def test_pa_scrape():
    pages = 10

    st_cpu = time.process_time()
    st = time.time()
    pa_listings = scrape_full_PA(PA_BASE_URL, pages=pages)
    print(f'cpu execution: {time.process_time() - st_cpu} sec')
    print(f'real execution: {time.time() - st} sec')
    print(f'{len(pa_listings)} total listings on {pages} pages')
    
    print('converting to df')

    df = pd.DataFrame(columns=[
            'id', 'title', 'price',
            'seller_name', 'url', 'tier', 
            'BE', 'server', 'total_orders',
            'member_since', 'stars', 'reviews', 'delivery'])

    for i, l in enumerate(pa_listings):
        df = pd.concat([df, l.to_pd_row()], ignore_index=True, sort=False)

    df = calculate_rank_from_pa_listing_text(df)
    df = calculate_server_from_pa_listing_text(df)

    return df

# print(test_pa_scrape())


def test_pa_scrape_v2():
    pages = 10

    st_cpu = time.process_time()
    st = time.time()
    pa_listings = scrape_full_PA_v2()
    print(f'cpu execution: {time.process_time() - st_cpu} sec')
    print(f'real execution: {time.time() - st} sec')
    print(f'{len(pa_listings)} total listings on {pages} pages')
    
    print('converting to df')

    df = pd.DataFrame(columns=[
            'id', 'title', 'price',
            'seller_name', 'url', 'tier', 
            'BE', 'server', 'total_orders',
            'member_since', 'stars', 'reviews', 'delivery'])

    for i, l in enumerate(pa_listings):
        df = pd.concat([df, l.to_pd_row()], ignore_index=True, sort=False)

    df = calculate_rank_from_pa_listing_text(df)
    df = calculate_server_from_pa_listing_text(df)

    return df

test_pa_scrape_v2()