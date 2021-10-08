import csv
import re


def zinc_processed_with_bracket(sen_space):
    all_smiles = []
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
        all_smiles.append(list(word))
    val = ["\n"]
    for i in range(len(all_smiles)):
        for j in range(len(all_smiles[i])):
            if all_smiles[i][j] not in val:
                val.append(all_smiles[i][j])

    return val, all_smiles


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


def smi_tokenizer(smi):
    """
    This function is based on https://github.com/pschwllr/MolecularTransformer#pre-processing
    Modified by Shoichi Ishida
    """
    pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    assert smi == ''.join(tokens)
    tokens.insert(0, '&')
    return tokens