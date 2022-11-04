from pyspark import SparkConf, SparkContext
import os
import sys
import random as rand

def wordsPerDoc(document, K=-1):
    pairs = {}
    words = document.split(" ")
    for word in words:
        pairs[word] = pairs.get(word,0) + 1
    if K==-1:
        return pairs.items()
    else:
        return [(rand.randint(0,K-1),(key, pairs[key])) for key in pairs.keys()]

def gather_pairs(pairs):
    pairs_dict = {}
    for pair in pairs[1]:
        word, count = pair[0], pair[1]
        pairs_dict[word] = pairs_dict.get(word,0)+count
    return pairs_dict.items()


def wordCount1(docs):
    word_count_pairs = (docs.flatMap(wordsPerDoc).reduceByKey(lambda x,y: x+y))
    return word_count_pairs

def wordCount2(docs, K):
    word_count_pairs = (docs.flatMap(lambda x: wordsPerDoc(x,K))
                        .groupByKey()
                        .flatMap(gather_pairs)
                        .reduceByKey(lambda x, y: x + y))
    return word_count_pairs

def wordCount3(docs, K):
    word_count_pairs = (docs.flatMap(lambda x: wordsPerDoc(x))
                        .groupBy(lambda x: (rand.randint(0,K-1)))
                        .flatMap(gather_pairs)
                        .reduceByKey(lambda x, y: x + y))
    return word_count_pairs



def main():
    assert len(sys.argv) == 3, "Usage: python WordCountExample.py <num_of_partitions> <file_name>"

    conf = SparkConf().setAppName("Myfirstcode").setMaster("local[*]")
    sc = SparkContext(conf=conf)

    K = sys.argv[1]
    assert K.isdigit(), "K must be an integer"
    K = int(K)

    data_path = sys.argv[2]
    assert os.path.isfile(data_path), "File or Folder not found"
    docs = sc.textFile(data_path, minPartitions=K).cache()
    docs.repartition(numPartitions=K)

    numdocs = docs.count()
    print("Number of documents = ",numdocs)

    print("number of distinct words in the documents using reduceByKey = ",wordCount1(docs).count())

    print("number of distinct words in the documents using groupByKey = ",wordCount2(docs,K).count())

    print("number of distinct words in the documents using groupBy = ", wordCount3(docs, K).count())


if __name__ == "__main__":
    main()