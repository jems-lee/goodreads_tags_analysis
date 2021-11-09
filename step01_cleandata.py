import numpy as np
import pandas as pd
from typing import Dict
import argparse


BAD_WORDS = {
    "read",
    "own",
    "my-",
    "my_",
    "default",
    "favorites",
    "favourites",
    "kindle",
    "book-club",
    "library",
    "audiobook",
    "audiobooks",
    "ebook",
    "to-buy",
    "series",
    "audio",
    "novels",
    "literature",
}
TAG_NAME_MAP = {
    "sci-fi": "science-fiction",
    "scifi": "science-fiction",
    "ya": "young-adult",
    "classic": "classics",
    "children": "childrens",
    "children-s": "childrens",
    "historical": "historical-fiction",
    "nonfiction": "non-fiction",
    "dystopia": "dystopian",
    "kids": "childrens",
    "memoirs": "memoir",
}
INCLUDE_TAGS = {
    "memoir",
    "science",
    "biography",
    "humor",
    "business"
}


def remove_tags_containing_bad_word(
    tags: pd.DataFrame, bad_tag_words: [str]
) -> pd.DataFrame:
    """Remove rows from book_tags that contain the strings in bad_tag_words"""
    filtered_tags = tags.copy()
    for tag in bad_tag_words:
        filtered_tags = filtered_tags[~filtered_tags.tag_name.str.contains(tag)]
    return filtered_tags.query("tag_id != 11743")


def convert_tag_name_map_to_tag_id_map(
    tag_name_map, tags: pd.DataFrame
) -> Dict[int, int]:
    tag_id_map = {}
    for from_word, to_word in tag_name_map.items():
        from_id = tags[tags.tag_name == from_word].tag_id.to_numpy()[0]
        to_id = tags[tags.tag_name == to_word].tag_id.to_numpy()[0]
        tag_id_map[from_id] = to_id
    return tag_id_map


def convert_tag_names_to_tag_ids(
    tag_names,
    tags):
    return tags.query("tag_name in @tag_names").tag_id


def consolidate_tags(book_tags, tag_id_map: Dict[str, str]) -> pd.DataFrame:
    cleaned_book_tags = book_tags.copy()
    for from_id, to_id in tag_id_map.items():
        cleaned_book_tags.loc[cleaned_book_tags.tag_id == from_id, "tag_id"] = to_id
    return cleaned_book_tags


def get_most_popular_n_tags(
    book_tags: pd.DataFrame, tags: pd.DataFrame, n: int
) -> pd.DataFrame:
    return (
        book_tags.merge(tags, on="tag_id")
        .groupby(["tag_id", "tag_name"], as_index=False)
        .sum()
        .sort_values("count", ascending=False)
        .head(n=n)
    )     


def pivot_book_tags_to_wide(book_tags, tags):
    return (
        book_tags.merge(tags, on="tag_id")
        .pivot_table(
            columns="tag_name", index="goodreads_book_id", values="count", fill_value=0
        )
        .assign(total=lambda d: d.sum(1))
        .apply(lambda x: x / x.total, axis=1)
        .drop(columns=["total"])
    )


def pivot_book_tags_to_wide_binary(
    book_tags: pd.DataFrame, tags: pd.DataFrame, cutoff=0.1
) -> pd.DataFrame:
    df = 1 * (pivot_book_tags_to_wide(book_tags, tags) > cutoff)
    return df


def pivot_book_tags_to_wide_log(book_tags, tags):
    return (
        book_tags.merge(tags, on="tag_id")
        .pivot_table(
            columns="tag_name", index="goodreads_book_id", values="count", fill_value=0
        )
        .apply(lambda x: np.log10(x+1))
    )


def merge_df_wide_with_books(df_wide, books):
    return books.filter(items=["book_id", "goodreads_book_id"]).merge(
        df_wide, on="goodreads_book_id"
    )


def get_all_relevant_book_tags(cleaned_book_tags, top_tag_ids, manual_tag_ids):
    return cleaned_book_tags.query("tag_id in @top_tag_ids or tag_id in @manual_tag_ids")


def create_book_data_csv(output_dir, book_tags, books, tags):
    """Create both book_data and book_data_binary csv."""
    df_wide = pivot_book_tags_to_wide(book_tags, tags)
    df_wide_binary = pivot_book_tags_to_wide_binary(book_tags, tags)
    df_wide_log = pivot_book_tags_to_wide_log(book_tags, tags)
    df = merge_df_wide_with_books(df_wide, books)
    df_binary = merge_df_wide_with_books(df_wide_binary, books)
    df_log = merge_df_wide_with_books(df_wide_log, books)
    df.to_csv(output_dir + "/book_data.csv", index=False)
    df_binary.to_csv(output_dir + "/book_data_binary.csv", index=False)
    df_log.to_csv(output_dir + "/book_data_log.csv", index=False)


def main():
    args = parse_args()
    output_dir = args.output_dir

    books = pd.read_csv("./goodbooks-10k/books.csv")
    book_tags = pd.read_csv("./goodbooks-10k/book_tags.csv")
    ratings = pd.read_csv("./goodbooks-10k/ratings.csv")
    tags = pd.read_csv("./goodbooks-10k/tags.csv")

    filtered_tags = remove_tags_containing_bad_word(tags, BAD_WORDS)
    tag_id_map = convert_tag_name_map_to_tag_id_map(TAG_NAME_MAP, filtered_tags)
    cleaned_book_tags = consolidate_tags(book_tags, tag_id_map)
    top_tags = get_most_popular_n_tags(cleaned_book_tags, filtered_tags, n=args.number_of_tags)
    filtered_book_tags = get_all_relevant_book_tags(
        cleaned_book_tags,
        top_tags.tag_id,
        convert_tag_names_to_tag_ids(INCLUDE_TAGS, tags)
    )
    create_book_data_csv(output_dir, filtered_book_tags, books, tags)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--number-of-tags", "-n", type=int, required=True)
    parser.add_argument("--output-dir", "-o", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main()
