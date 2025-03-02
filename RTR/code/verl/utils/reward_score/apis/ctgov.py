import json
import requests
import traceback
import copy

import pandas as pd



CTGOV_BASE_URL = "https://clinicaltrials.gov/api/v2/studies?query.term="
CTGOV_NCTID_BASE_URL = "https://clinicaltrials.gov/api/v2/studies?&query.id="


class CTGovAPI:
    """A wrapper class for the CT.gov API.
    """
    def __init__(self, retry=1):
        self.retry = retry

    def search(self, query, topk=-1):
        """Get the response of nctids."""
        search_results = self.search_with_query(query, topk=topk)

        return search_results

    def search_with_query(self, query, topk=-1):
        """Search with query input to get the response of nctids."""
        err_msg = ""

        if "pageSize=" not in query:
            query += "&pageSize=1000" # max of api is 1000

        nctid_list = []
        for i in range(self.retry):
            if i > 0:
                print(f"Retry {i} times")
            try:
                response = requests.get(query)
                if response.status_code != 200:
                    raise ConnectionError(f"CTGov connection error occurred - {response.text}")

                # parse the response to format a list of trials to display
                output_df = self._parse_response(response.text, query, topk)
                if len(output_df) == 0:
                    nctid_list = []
                else:
                    nctid_list = output_df['NCT Number'].tolist()[:topk]

                break
            except:
                err_msg = traceback.format_exc()
                print(err_msg)

        if err_msg != "":
            raise RuntimeError("A CTGOV API error occurred")

        search_results = nctid_list
        return search_results

    def search_with_keywords(self, keywords: str, topk=-1):
        """Search with keywords input to get the response of nctids."""
        query = CTGOV_BASE_URL + keywords
        search_results = self.search_with_query(query, topk=topk)

        return search_results
    
    def get_trials_by_nctids(self, nctid_list):
        """Search nctids to get the summary of trials."""
        err_msg = ""
        trials = None

        for i in range(self.retry):
            if i > 0:
                print(f"Retry {i} times")
            try:
                nctid_list_str = " OR ".join(list(set(nctid_list)))
                query = CTGOV_NCTID_BASE_URL + nctid_list_str + "&pageSize=1000"
                response = requests.get(query)
                if response.status_code != 200:
                    raise ConnectionError(f"CTGOV connection error occurred - {response.text}")

               # parse the response to format a list of trials to display
                trials = self._parse_response(response.text, query)

                break
            except:
                err_msg = traceback.format_exc()
                print(err_msg)

        return trials

    def _parse_response(self, response, query, topk=-1):
        """Parse the response to pd dataframe from the API call."""
        # achieve maximum retrieval number
        if topk != -1 and topk <= 0:
            return []
        
        results = json.loads(response)
        total_count = results.get("totalCount", 0)
        study_results = results.get("studies", {})
        nextPageToken = results.get("nextPageToken", "")

        studies = []
        for res in study_results:
            res = self._parse_json_response(res)
            studies.append(res)

        if studies != []:
            studies = pd.concat(studies, axis=0).reset_index(drop=True)
        else:
            return pd.DataFrame()

        # recursion to get results from all pages
        if nextPageToken != "":
            query_next_page = query + f"&pageToken={nextPageToken}"
            response_next_page = requests.get(query_next_page)
            if topk == -1:
                studies_next_page = self._parse_response(response_next_page.text, query, topk=-1)
            else:
                studies_next_page = self._parse_response(response_next_page.text, query, topk=topk-len(studies))

            if isinstance(studies_next_page, pd.DataFrame):
                studies = pd.concat([studies, studies_next_page], ignore_index=True)

        return studies


    def _parse_json_response(self, response: dict):
        """Parse important part of the retrieved json trial info from CT.gov.
        """
        data = response.get("protocolSection", {})
        nct_number = data.get('identificationModule', {}).get('nctId', None)
        study_title = data.get('identificationModule', {}).get('briefTitle', None)
        brief_summary = data.get('descriptionModule', {}).get('briefSummary', None)


        # Create a DataFrame
        df = pd.DataFrame({
            'NCT Number': [nct_number],
            'Study Title': [study_title],
            'Brief Summary': [brief_summary],
        })

        return df
