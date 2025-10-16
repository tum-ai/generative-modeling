import csv


def convert_cmu_book_summaries(input_path: str, output_path: str):
    with (
        open(input_path, "r", encoding="utf-8") as fin,
        open(output_path, "w", encoding="utf-8") as fout,
    ):
        reader = csv.reader(fin, delimiter="\t")

        for row in reader:
            if len(row) < 7:
                continue
            title = row[2].strip()
            summary = row[6].strip()

            if not title or not summary:
                continue

            fout.write(f"<BOS>{title}<SUM>{summary}<EOS>\n")


def main():
    convert_cmu_book_summaries(
        "data/booksummaries.txt", "data/booksummaries_preprocessed.txt"
    )


if __name__ == "__main__":
    main()
