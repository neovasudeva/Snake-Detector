import requests
import time
import wget
from selenium import webdriver
from timeout import timeout
import os

# function that fetches images given some search URL
def fetch_image_urls(image_urls, search_url, max_links_to_fetch, wd, sleep_between_interactions=1):
    """
    INPUTS: 
        - image_urls: a set containing image_urls
        - search_url: Google image URL to scrape images from
        - max_links_to_fetch: number of images to scrape
        - webdriver: Selenium web driver to use
        - sleep_between_interactions: sleep time to not blow up the program 
    OUTPUTS: 
        - The image_urls set will have image urls added in 
    """

    # function that scrolls to bottom of page
    def scroll_to_end(wd):
        wd.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(sleep_between_interactions)    

    # load the page
    wd.get(search_url)

    # record results found
    results_start = 0
    images_added = 0

    # loop through every image
    #len(image_urls)
    while images_added < max_links_to_fetch:
        scroll_to_end(wd)

        # get all image thumbnail results
        thumbnail_results = wd.find_elements_by_css_selector("img.Q4LuWd")
        number_results = len(thumbnail_results)
        
        # for every thumbnail result, click image and obtain actual image URL
        for img in thumbnail_results[results_start:number_results]:
            # try to click every thumbnail such that we can get the real image behind it
            try:
                img.click()
                time.sleep(sleep_between_interactions)
            except Exception:
                continue

            # extract image urls and add them to set 
            actual_images = wd.find_elements_by_css_selector('img.n3VNCb')
            for actual_image in actual_images:
                if actual_image.get_attribute('src') and 'http' in actual_image.get_attribute('src'):
                    image_urls.add(actual_image.get_attribute('src'))
                    images_added += 1

            # if number of image URLs pulled is enough, break
            if images_added >= max_links_to_fetch:
                print(f"Found: {len(image_urls)} image links in total have been added!")
                break
            # else keep looking for more images
            else:
                print("Found:", images_added, "image links, looking for more ...")
                
                # if we've reached limit of page, click "load more results" button
                load_more_button = wd.find_element_by_css_selector(".mye4qd")
                if load_more_button:
                    wd.execute_script("document.querySelector('.mye4qd').click();")

        # move the result startpoint further down
        results_start = len(thumbnail_results)

# define wget into a wrapper function for a timeout of 5 seconds
@timeout(5)
def download(url, location):
    wget.download(url, location)

# launch Chrome with Selenium and fetch images with 
wd = webdriver.Chrome("./bin/chromedriver")

# list of URLs to pull images from 
google_urls = ["https://www.google.com/search?q=snakes+in+georgia&tbm=isch&ved=2ahUKEwi36tH6gNvpAhVNBFMKHfduAe8Q2-cCegQIABAA&oq=snakes+in+georgia&gs_lcp=CgNpbWcQA1AAWABgms43aABwAHgAgAEAiAEAkgEAmAEAqgELZ3dzLXdpei1pbWc&sclient=img&ei=lwPSXvfCNs2IzAL33YX4Dg&bih=949&biw=1853", "https://www.google.com/search?q=snake+in+grass&source=lnms&tbm=isch&sa=X&ved=2ahUKEwit-Jf5m9zpAhXTVs0KHUnhCYMQ_AUoAXoECB4QAw&biw=1853&bih=949", "https://www.google.com/search?q=copperhead&source=lnms&tbm=isch&sa=X&ved=2ahUKEwiCltyInNzpAhURHc0KHcEBDvoQ_AUoAXoECBsQAw&biw=1853&bih=949", "https://www.google.com/search?q=cottonmouth&source=lnms&tbm=isch&sa=X&ved=2ahUKEwih--2jnNzpAhWBJ80KHVfQA1QQ_AUoAXoECBkQAw&biw=1853&bih=949", "https://www.google.com/search?q=garter+snake&source=lnms&tbm=isch&sa=X&ved=2ahUKEwiniuO5nNzpAhWIW80KHe0qDOEQ_AUoAXoECBcQAw&biw=1853&bih=949", "https://www.google.com/search?q=snakes+in+desert&tbm=isch&ved=2ahUKEwjknp-9nNzpAhWFBFMKHZ7DDvUQ2-cCegQIABAA&oq=snakes+in+desert&gs_lcp=CgNpbWcQAzICCAAyBggAEAUQHjIGCAAQBRAeMgYIABAFEB4yBggAEAUQHjIGCAAQBRAeMgYIABAIEB4yBggAEAgQHjIGCAAQCBAeMgYIABAIEB46BAgAEENQo7QBWLPAAWClwQFoAHAAeACAAXKIAbQMkgEDOS43mAEAoAEBqgELZ3dzLXdpei1pbWc&sclient=img&ei=q6bSXqSdBIWJzAKeh7uoDw&bih=949&biw=1853"]

# define number of images to pull from each Google Images URL and begin search
num_images = 0
image_urls = set()
count = 1
for url in google_urls:
    print("Fetching images from URL #", count)
    fetch_image_urls(image_urls, url, num_images, wd)
    print("\n")
    count += 1

# close web driver
wd.close()

# download each photo into "training" folder
url_count = 0
fails = 0
for url in image_urls:
    # manipulate URL to get get only the image
    index = url.rfind(".jpg")
    short_url = url if index == -1 else url[ : index + 4]
    print(short_url + "\n")

    # download image
    try: 
        download(short_url, "./training/" + str(url_count) + ".jpg")
    except Exception:
        print("Download of image with id " + str(url_count) + " failed or timed out.")
        fails += 1

    print("\n")
    url_count += 1

# update fail rate
try: 
    print("Success rate: ", str((num_images * len(google_urls) - fails) / (num_images * len(google_urls))))
except Exception:
    print("Download success rate calculation failed.")

