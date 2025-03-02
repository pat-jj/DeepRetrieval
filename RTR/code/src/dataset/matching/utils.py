

def read_trec_qrels(trec_qrel_file):
    '''
    Read a TREC style qrel file and return a dict:
        QueryId -> docName -> relevance
    '''
    qrels = []
    with open(trec_qrel_file) as fh:
        for line in fh:
            try:
                query, zero, doc, relevance = line.strip().split()
                qrels.append((query, doc, int(relevance)))
            except Exception as e:
                print ("Error: unable to split line in 4 parts", line)
                raise e
    return qrels