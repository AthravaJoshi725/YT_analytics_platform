import os
import logging
import config
import time
from cachetools import TTLCache

from dotenv import load_dotenv
import pandas as pd
import googleapiclient.discovery

from queue import Queue
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()
API_KEY = os.getenv("API_KEY")


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.OUTPUT_PATHS['log_file']),
        logging.StreamHandler()
    ]
)


# cache upto 10 video comments for 30 minutes
yt_comments_cache = TTLCache(maxsize=10, ttl = 1800)
yt_details_cache = TTLCache(maxsize=10, ttl=1800)

def extract_video_id(yt_link):
    # This function takes yt_link and extracts video ID from it
    '''
    https://www.youtube.com/watch?v=JeVbPoDzLq8     TYPE_1
    https://www.youtube.com/watch?v=XxCAcJv4VyE&t=1040s
    https://youtu.be/JeVbPoDzLq8?si=SpkP9XeFXoPkvX2J   TYPE_2
    https://youtu.be/XxCAcJv4VyE?si=hBrd14x9yh6CjCvr
    '''
    if "youtube.com/watch?v=" in yt_link:
        video_id = yt_link.split("watch?v=")[1].split("&")[0]
        logging.info(f"Extracted videoID: {video_id} from link of TYPE_1: {yt_link}")
    elif "youtu.be/" in yt_link:
        video_id = yt_link.split("youtu.be/")[1].split("?")[0]
        logging.info(f"Extracted videoID: {video_id} from link of TYPE_2: {yt_link}")
    else:
        logging.error(f"Invalid YouTube link format: {yt_link}")
        return None

    return video_id

def extract_video_detail(video_id):
    # check cache
    if video_id in yt_details_cache:
        logging.info(f"Loading details from cache: {video_id}")
        return yt_details_cache[video_id]


    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"
    api_service_name = "youtube"
    api_version = "v3"
    DEVELOPER_KEY = API_KEY
    try:
        youtube = googleapiclient.discovery.build(api_service_name, api_version, developerKey = DEVELOPER_KEY)

        request = youtube.videos().list(
            part="snippet,statistics",
            id=video_id
        )
        response = request.execute()
        logging.info(f"yt_details extracted for {video_id} ")

    except Exception as e:
        logging.error(f"Error while getting yt_details response: {e}")
        return {}


    # Parse response
    video_details = {}
    try:
        item = response["items"][0]['snippet']
        video_details = {
            "title": item.get("title"),
            "channelName": item.get("channelTitle"),
            "description": item.get("description"),
            "publishedAt": item.get("publishedAt")
            }
        
        yt_details_cache[video_id] = video_details
        logging.info(f"{video_id} details saved in cache")

        logging.info(f"Parsed video details for video_id: {video_id}")
        return video_details
    
    except Exception as e:
        logging.error("Parsing for yt_details failed")
        return {}
    


def extract_comments(video_id):

    # chech cache first using video id
    if video_id in yt_comments_cache:
        logging.info(f"Loading comments from cache: {video_id}")
        return {"items": yt_comments_cache[video_id]}
    

    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"
    api_service_name = "youtube"
    api_version = "v3"
    DEVELOPER_KEY = API_KEY

    all_comments = []
    seen_tokens = set() # to avoid duplicates
    token_queue = Queue() 
    lock = threading.Lock()
     
    try:
        youtube = googleapiclient.discovery.build(
            api_service_name, api_version, developerKey = DEVELOPER_KEY)
        
        
        # fetch one page with the next page token
        def fetch_page(token = None):
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=100,
                pageToken = token,
                textFormat = "plainText"
            )
            return request.execute()

        # collect the first page
        first_page = fetch_page()
        all_comments = first_page.get("items", [])
        next_token = first_page.get('nextPageToken')

        
        if next_token:
            token_queue.put(next_token)
            seen_tokens.add(next_token)

        def worker():
            while True:
                try:
                    token = token_queue.get(timeout=1)
                except:
                    return
                
                try:
                    response = fetch_page(token)
                    items = response.get("items", [])
                    new_token = response.get("nextPageToken")
                    
                    with lock:
                        all_comments.extend(items)

                    with lock:
                        if new_token and new_token not in seen_tokens:
                            token_queue.put(new_token)
                            seen_tokens.add(new_token)

                except Exception as e:
                    logging.error(f"[ERROR] Failed to fetch page token {token}")

                finally:
                    token_queue.task_done()

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(worker) for _ in range(5)]
            for _ in as_completed(futures):
                pass
        
        token_queue.join()
        
        # store in cache
        yt_comments_cache[video_id] = all_comments
        logging.info(f'Fetched {len(all_comments)} comments for video ID {video_id}')

    except Exception as e:
        logging.error(f'Error while fetching comments for video ID {video_id}: {e}')
        return {"items": []}

    return {"items": all_comments}


def parse_comments(response):
    comments_data = []
    try:
        for item in response.get("items",[]):
            snippet = item['snippet']['topLevelComment']['snippet']
            comments_info = {
                "author": snippet.get("authorDisplayName"),
                "comment": snippet.get("textOriginal"),
                "likes": snippet.get("likesCount"),
                "published_at": snippet.get("publishedAt"),
            }
            comments_data.append(comments_info)
        logging.info(f"Parsed {len(comments_data)} comments from response and converted to dataframe.")
        df = pd.DataFrame(comments_data)

    except Exception as e:
        logging.error(f'Error while parsing comments: {e}')

    return df

def func_get_comments(video_id):
    start = time.time()
    # video_id = extract_video_id(video_link)
    # ytDetails_response = extract_video_detail(video_id)
    comments_response = extract_comments(video_id)
    data = parse_comments(comments_response)
    end = time.time()

    logging.info(f'Comment Extraction [{video_id}]: Time taken: {end- start:.2f}s')
    return data

def main():
    # user_input = input("Enter the youtube video link: ")
    user_input = 'https://youtu.be/j5168Ug7DvA?si=PdAny4VQyKW3ry89'
    video_id = extract_video_id(user_input)
    response = extract_comments(video_id)
    data= parse_comments(response)
    yt_details = extract_video_detail(video_id)
    print(data)
    print(yt_details)

if __name__ == "__main__":
    main()