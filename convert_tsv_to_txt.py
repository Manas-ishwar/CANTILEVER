import csv

# Convert movie_lines.tsv to movie_lines.txt
with open("data/movie_lines.txt.txt", "r", encoding="utf-8", errors="ignore") as tsv_file, \
     open("data/movie_lines.txt", "w", encoding="utf-8") as txt_file:
    tsv_reader = csv.reader(tsv_file, delimiter="\t")
    for row in tsv_reader:
        if len(row) >= 5:
            txt_file.write(" +++$+++ ".join(row[:5]) + "\n")

# Convert movie_conversations.tsv to movie_conversations.txt
with open("data/movie_conversations.txt.txt", "r", encoding="utf-8", errors="ignore") as tsv_file, \
     open("data/movie_conversations.txt", "w", encoding="utf-8") as txt_file:
    tsv_reader = csv.reader(tsv_file, delimiter="\t")
    for row in tsv_reader:
        if len(row) >= 4:
            txt_file.write(" +++$+++ ".join(row[:4]) + "\n")

print("Conversion completed successfully.")
