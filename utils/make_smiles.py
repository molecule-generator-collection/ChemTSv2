import csv


def zinc_processed_with_bracket(sen_space):
    all_smile = []
    length = []
    end = "\n"
    element_table = ["C", "N", "B", "O", "P", "S", "F", "Cl", "Br", "I", "(", ")", "=", "#"]

    for i in range(len(sen_space)):
        word_space = sen_space[i]
        word = []
        j = 0
        while j < len(word_space):
            word_space1 = []
            if word_space[j] == "[":
                word_space1.append(word_space[j])
                j = j + 1
                while word_space[j] != "]":
                    word_space1.append(word_space[j])
                    j = j + 1
                word_space1.append(word_space[j])
                word_space2 = ''.join(word_space1)
                word.append(word_space2)
                j = j + 1
            else:
                word_space1.append(word_space[j])

                if j + 1 < len(word_space):
                    word_space1.append(word_space[j+1])
                    word_space2 = ''.join(word_space1)
                else:
                    word_space1.insert(0, word_space[j-1])
                    word_space2 = ''.join(word_space1)

                if word_space2 not in element_table:
                    word.append(word_space[j])
                    j = j + 1
                else:
                    word.append(word_space2)
                    j = j + 2

        word.append(end)
        word.insert(0, "&")
        len1 = len(word)
        length.append(len1)
        all_smile.append(list(word))
    val = ["\n"]
    for i in range(len(all_smile)):
        for j in range(len(all_smile[i])):
            if all_smile[i][j] not in val:
                val.append(all_smile[i][j])

    return val, all_smile


def zinc_data_with_bracket_original(file_dir):
    sen_space = []
    f = open(file_dir, 'r')

    reader = csv.reader(f)
    for row in reader:
        sen_space.append(row)
    f.close()

    word1 = sen_space[0]

    zinc_processed = []
    for i in range(len(sen_space)):
        word1 = sen_space[i]
        zinc_processed.append(word1[0])

    return zinc_processed
