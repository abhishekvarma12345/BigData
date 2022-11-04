from pyspark import SparkConf, SparkContext
import os
import sys
import random as rand

def Posquantcountry(row, S):
    fields = row.split(",")
    country = fields[7]
    quantity = int(fields[3])
    if (country == S or S == "all") and quantity>0:
        return True
    else:
        return False

def ProductCustomer(row):
    fields = row.split(",")
    productID = fields[1]
    customerID = int(fields[6])
    return [((productID, customerID),rand.randint(0,10))]   

def GatherProductPairs(pairs):
    pairs_dict = {}
    for p in pairs:
        prodcust, val = p[0], p[1]
        pairs_dict[prodcust[0]] = pairs_dict.get(prodcust[0], 0) + 1
    return pairs_dict.items()

def ProductCounts(pair):
    prodcust = pair[0]
    return [(prodcust[0],1)]

def reverse(pair):
    return [(pair[1],pair[0])]




def main():
    # check for correct no of arguments
    assert len(sys.argv) == 5, "Usage: python ppl.py <K:num_of_partitions> <H:no of highest popularity products> <S:country name> <file_name>"

    # setting up the configuration
    conf = SparkConf().setAppName("ProductPopularity").setMaster("local[*]")
    sc = SparkContext(conf=conf)

    # no of partitions
    K = sys.argv[1]
    assert K.isdigit(), "K must be an integer" # check for K is an integer
    K = int(K)

    H = sys.argv[2]
    assert H.isdigit(), "H must be an integer" # check for H is an integer
    H = int(H)

    # S must be a string
    S = sys.argv[3]
    

    data_path = sys.argv[4]
    assert os.path.isfile(data_path), "File or Folder not found" # check if file exists

    rawData = sc.textFile(data_path, minPartitions=K).cache()
    rawData.repartition(numPartitions=K)

    print("Number of rows = ",rawData.count()) 

    productCustomer = (rawData.filter(lambda x: Posquantcountry(x,S)) 
                       .flatMap(ProductCustomer)
                       .groupByKey()
                       .reduceByKey(lambda x,y: x + y))

    print("Product-Customer Pairs = ",productCustomer.count())

    ProductPopularity1 = (productCustomer.mapPartitions(GatherProductPairs)
                          .groupByKey()
                          .mapValues(lambda vals: sum(vals)))

    ProductPopularity2 = (productCustomer.flatMap(ProductCounts)
                          .reduceByKey(lambda x, y: x + y))
    
    if H == 0:
        lst1 = ProductPopularity1.sortByKey().collect()
        print("productPopularity1:")
        for prod, popularity in lst1:
            print(f"Product: {prod} Popularity: {popularity};",end=" ")
        print()
        lst2 = ProductPopularity2.sortByKey().collect()
        print("productPopularity2:")
        for prod, popularity in lst2:
            print(f"Product: {prod} Popularity: {popularity};",end=" ")
        print()
    else:
        lst = ProductPopularity1.flatMap(reverse).sortByKey(ascending=False).take(H)
        print(f"Top {H} Products and their Popularities")
        for popularity, prod in lst:
            print(f"Product {prod} Popularity {popularity};",end=" ")
        print()



if __name__ == "__main__":
    main()