def get_frequency_vocab(path):

    file=open(path, "r")

    vocab_words={}
    vocab_labels={}

    for line in file.readlines():

        line=line.strip("\n").split(",")

        words=line[0]
        label=line[1].strip()

        words=words.split()

        for word in words:

            if word in vocab_words.keys():

                vocab_words[word]=vocab_words[word]+1
            else:

                vocab_words[word]=1

        if label in vocab_labels.keys():

            vocab_labels[label]=vocab_labels[label]+1
        else:
            vocab_labels[label]=1

    return vocab_words, vocab_labels

