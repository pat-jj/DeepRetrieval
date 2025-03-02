import json
import requests
import traceback

import pandas as pd
import xml.etree.ElementTree as ET



PUBMED_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term="
PUBMED_SUMMARY_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&id="
PUBMED_EFETCH_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id="


class PubmedAPI:
    """A wrapper class for the Pubmed API.
    """
    def __init__(self, retry=1, api_key=None):
        self.retry = retry
        self.api_key = api_key

    def search(self, query, topk=-1):
        """Get the response of pmids."""
        search_results = self.search_with_query(query, topk=topk)

        return search_results

    def search_with_query(self, query, topk=-1):
        """Search with query input to get the response of pmids."""
        err_msg = ""
        if self.api_key:
            query += f"&api_key={self.api_key}"
        if '&retmode=json' not in query:
            query += '&retmode=json'

        retstart = 0
        retmax = 3000
        query += f'&retmax={retmax}'

        for i in range(self.retry + 1):
            if i > 0:
                print(f"Retry {i} times")
            try:
                response = requests.get(query)
                # response_text = response.text
                
                if response.status_code != 200:
                    raise ConnectionError(f"Pubmed connection error occurred - {response.text}")
                response_dict = json.loads(response.text)
                pmid_list = []
                count = int(response_dict['esearchresult']['count'])
                # 'retstart' cannot be larger than 9998 in esearch
                while retstart < count and (retstart < topk or topk == -1) and retstart <= 9998:
                    response = requests.get(query + f'&retstart={retstart}')
                    if response.status_code != 200:
                        raise ConnectionError(f"Pubmed connection error occurred - {response.text}")
                    response_dict = json.loads(response.text)
                    pmid_list.extend(response_dict['esearchresult']['idlist'])
                    
                    retstart += retmax

                pmid_list = list(set(pmid_list))[:topk]

                break
            except:
                err_msg = traceback.format_exc()
                print(err_msg)

        if err_msg != "":
            raise RuntimeError("A Pubmed API error occurred")

        search_results = pmid_list
        return search_results
    
    def search_with_keywords(self, keywords: str, topk=-1):
        """Search with keywords input to get the response of pmids."""
        query = PUBMED_BASE_URL + keywords
        search_results = self.search_with_query(query, topk=topk)

        return search_results
    
    def get_papers_by_pmids(self, pmid_list):
        """Search pmids to get the summary of paper."""
        err_msg = ""
        for i in range(self.retry + 1):
            if i > 0:
                print(f"Retry {i} times")
            try:
                # build the query to get the summary of the articles
                pmid_list_str = ','.join(pmid_list)
                summary_query = PUBMED_SUMMARY_BASE_URL + pmid_list_str + "&retmode=json"
                if self.api_key:
                    summary_query += f"&api_key={self.api_key}"
                response = requests.get(summary_query)

                if response.status_code != 200:
                    if response.status_code == 414:
                        raise ConnectionError(f"Pubmed query too long!")
                    else:
                        raise ConnectionError(f"Pubmed connection error occurred: {response.text}")

                response = json.loads(response.text)
                results = response.get("result", {})
                uids = results.get("uids", [])

                if len(uids) == 0:
                    return pd.DataFrame()
                
                # build the query to parse the remaining sections
                papers = []
                for idx in range(len(uids)):
                    try:
                        cur_res = results[uids[idx]]
                        parse_res = self._parse_json_summary_response(cur_res)
                        papers.append(parse_res)
                    except:
                        print(traceback.format_exc())
                papers = pd.concat(papers, axis=0).reset_index(drop=True)
                
                # retrieve abstract from efetch API
                abstract, mesh_term = self._retrieve_abstract_and_mesh_term_by_efetch(uids)
                papers['Abstract'] = abstract
                papers['Mesh Term'] = mesh_term

                break
            except:
                err_msg = traceback.format_exc()
                print(err_msg)
        
        if err_msg != "" or len(pmid_list) == 0:
            raise RuntimeError("A Pubmed API error occurred")

        return papers
    
    def _parse_json_summary_response(self, res):
        """Parse the summary json response to pd dataframe from the API call."""
        pmid = res.get("uid", None)
        pub_date = res.get("pubdate", None)
        journal = res.get("fulljournalname", None)
        if journal is None:
            journal = res.get("source", None)
        title = res.get("title", None)
        authors = res.get("authors", [])
        authors = [author.get("name", None) for author in authors]
        authors = "; ".join(authors)
        volume = res.get("volume", None)
        issue = res.get("issue", None)
        pages = res.get("pages", None)
        pubtypes = res.get("pubtype", [])
        url = 'https://pubmed.ncbi.nlm.nih.gov/' + pmid

        df = pd.DataFrame({
            "PMID": [pmid],
            "Publication Date": [pub_date],
            "Title": [title],
            "Authors": [authors],
            "Journal": [journal],
            "Volume": [volume],
            "Issue": [issue],
            "Pages": [pages],
            "Pubtypes": [pubtypes],
            "URL": [url]
        })
        return df
    
        
    def _retrieve_abstract_and_mesh_term_by_efetch(self, pmid_list):
        '''Retrieve abstract based on pmids from efetch API'''
        pmid_list_str = ','.join(pmid_list)
        query = PUBMED_EFETCH_BASE_URL + pmid_list_str + "&retmode=xml"
        if self.api_key:
            query += "&api_key=" + self.api_key

        response = requests.get(query)
        if response.status_code != 200:
            return [""] * len(pmid_list), [[]] * len(pmid_list)
        
        abstracts = {}
        mesh_terms_dict = {}
        response = response.text
        tree = ET.ElementTree(ET.fromstring(response))
        articles = tree.findall(".//PubmedArticle")
        for article in articles:
            abstract = article.find(".//AbstractText")
            if abstract is not None:
                abstract = abstract.text
            else:
                abstract = ""

            # get pmid
            article_ids = [a for a in article.findall(".//ArticleId") if a.get("IdType").lower() == "pubmed"]
            if len(article_ids) > 0:
                pmid = article_ids[0].text
                abstracts[pmid] = abstract
            else:
                continue
        
            # Get MeSH terms and Qualifiers
            mesh_terms = []
            mesh_headings = article.findall(".//MeshHeading")
            for mesh_heading in mesh_headings:
                descriptor = mesh_heading.find("DescriptorName")
                if descriptor is not None:
                    qualifiers = mesh_heading.findall("QualifierName")
                    if qualifiers:
                        for qualifier in qualifiers:
                            mesh_terms.append(f"{descriptor.text} ({qualifier.text})")
                    else:
                        mesh_terms.append(descriptor.text)

            abstracts[pmid] = abstract
            mesh_terms_dict[pmid] = mesh_terms
        
        # Make sure the input and output are in the same order and length
        output_abstracts = []
        output_mesh_terms = []
        for pmid in pmid_list:
            output_abstracts.append(abstracts.get(pmid, ""))
            output_mesh_terms.append(mesh_terms_dict.get(pmid, []))

        return output_abstracts, output_mesh_terms
        
