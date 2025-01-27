import pandas as pd
import re

def process_reviews(file_content, score):
    """
    Process review content and create a DataFrame with reviews and scores.
    
    Parameters:
    file_content (str): Content of the review file
    score (int): Review score to assign to all reviews in this file
    
    Returns:
    pandas.DataFrame: DataFrame with 'review_text' and 'score' columns
    """
    # Split the content into individual reviews
    reviews = file_content.split("==================================================\n")
    
    # Process each review
    processed_reviews = []
    
    for review in reviews:

        review = review.strip()

        #Remove Review #: tag. This info was only for Human review but should not be included in the review text itself
        review = re.sub(r'^Review \d+:\n', '', review)

        # Skip empty reviews or reviews that only contain location/address info
        if not review.strip() or any(x in review.lower() for x in ['1924 pennsylvania ave', 'washington, dc 20006', 'miles away from founding farmers -']):
            continue
            
        # Skip thank you responses from staff
        if review.lower().startswith('thank you') or review.lower().startswith('hi ') or review.lower().startswith('hi,'):
            continue
            
        # Add the review and score to our list
        processed_reviews.append({
            'review_text': review.strip(),
            'score': score
        })
    
    # Convert to DataFrame
    return pd.DataFrame(processed_reviews)

def main():
    # Process reviews from both files
    reviews_1 = process_reviews(open("yelp_reviews_1_FoundingFarmers.txt").read(), 1)
    reviews_2 = process_reviews(open("yelp_reviews_2_FoundingFarmers.txt").read(), 2)
    reviews_3 = process_reviews(open("yelp_reviews_3_FoundingFarmers.txt").read(), 3)
    reviews_4 = process_reviews(open("yelp_reviews_4_FoundingFarmers.txt").read(), 4)
    reviews_5 = process_reviews(open("yelp_reviews_5_FoundingFarmers.txt").read(), 5)
    
    # Combine the DataFrames
    all_reviews = pd.concat([reviews_1, reviews_2, reviews_3, reviews_4, reviews_5], ignore_index=True)
    
    # Display the first few reviews and basic statistics
    print("\nFirst few reviews:")
    print(all_reviews.head())
    
    print("\nReview count by score:")
    print(all_reviews['score'].value_counts())
    
    return all_reviews

if __name__ == "__main__":
    df = main()
    df.to_csv("Scraped_Data.csv", index=False)