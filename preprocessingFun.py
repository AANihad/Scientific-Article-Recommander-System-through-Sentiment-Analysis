import re
from Reference import Reference
from refextract import extract_references_from_file

CITATION_TYPE = ''# '[NUMBER]' or  'AUTHOR_YEAR'
months = "january|february|march|april|may|june|july|august|september|october|november|december"


def cleanReference(text):
    text = text.replace('-\n', '')
    text = text.replace('-', '')
    text = text.replace('\n', '')
    
    # print(text)
    return text
    # text =  re.sub(r'\n\n', '\n-#-\n', text).replace('\n', '')
    
    # text = text.replace('“', '"').replace('”', '"').replace('„', '"').replace('‟', '"').replace('–',
    #  '-').replace('-', '').replace('et al.', 'et al').replace('-\n', '').replace(' ', '\s').replace('\\n', '')

    # l = str(text).split('-#-')
    # return l
    
def findRef (keys, references):
    global CITATION_TYPE
    # print('CitType '+CITATION_TYPE)
    # print(keys)
    refs = []
    # print(keys)
    for ref in references:
        refType = False # if the author references only author name or author + year
        # print("--------------------------------------------------------------------------------------------------")
        # ref = ref.replace('[', '').replace(']', '')
        rt = ref.getText().replace(',', '').replace('.', '')
        for key in keys:
            if (CITATION_TYPE == "[NUMBER]"):
                if all(ref.getText().startswith(k) for k in key ):
                    # print(ref)
                    refs.append(ref)
                    refType = True
            elif(CITATION_TYPE == 'AUTHOR_YEAR'):
                if all( k in rt for k in key ):
                    # print('all of'+ str(key)+' in '+rt)
                    refs.append(ref)
                    refType = True
                    break
                # else :
                #     print('Strong Search failded for '+str(key)+' in '+rt)

        if not refType:
            for key in keys:
                # print("starting weak search for "+ str(keys))
                for k in key:
                    if (not re.search('\d{4}', k)) and (k in rt):
                        refs.append(ref)
        if refType:
            break
    # print('**************')
    # print(len(refs))
    return refs


def clean_string(text):
    final_string = ""

    # from https://stackoverflow.com/questions/3809401/what-is-a-good-regular-expression-to-match-a-url
    urlRegEx = "(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})"

    # Remove puncuation
    text = text.replace('“', '"').replace('”', '"').replace('„', '"').replace('‟', '"').replace('–',
     '-').replace('-', '').replace('et al.', 'et al').replace('-\n', '').replace(' ', '\s').replace('\\n', '').replace('\\r', '')

    # Remove line breaks
    text = re.sub(r'\n', ' ', text)

    # Remove stop words
    text = text.split()
    # useless_words = nltk.corpus.stopwords.words("english")
    # text_filtered = [word for word in text if not word in useless_words]

    final_string = ' '.join(text)

    return final_string


def containsCitation(token):
    square = "(\[\d+\](,\s?\[\d+\])*)"
    # [65]    # [89], [12]

    squareMultiple = "(\[\d+((,|-)\s?\d+)*\])"
    # [32-98]    # [87, 56]

    # etAl = "\(?(\b(?!(?:"+months+"))\w+ ((et al(\.|\,)?)|(and \w+)*)*(\,?\s*\(?\d{4}[a-z]?\)?)\;?)+\)?"
    etAl = "\(?(\w+ ((et al(\.|\,)?)|(and \w+)*)*(\,?\s*\(?\d{4}[a-z]?\)?)\;?)+\)?"
    # (nye et al, 2018)
    # it will not match june 2004

    parenthesis = "(\(\w+ \w* (et al\.?)?\s\d{4}[a-z]?((\,|\;) (\w+ )*\d{4}[a-z]?)*\))"
    # (Küçüktunç etal. 2012a, b, 2013c, 2015).
    # (Sesagiri Raamkumar et al. 2015, 2016a; Sesagiri Raamkumar and Foo 2018)

    regexp = re.compile(square+'|'+squareMultiple+'|'+etAl+'|'+parenthesis)
    return regexp.findall(token)


def Filter(list1, list2):
    return list((filter(lambda val: val not in list2, list1)))


def findKeys(citation):
    global CITATION_TYPE
    square =re.compile("(\[\d+\](,\s?\[\d+\])*)")
    # [65]    # [89], [12]
    squareMultiple =re.compile("(\[\d+((,|-)\s?\d+)*\])")
    # [32-98]    # [87, 56]
    
    # etAl = re.compile("\(?(\b(?!(?:"+months+"))\w+ ((et al(\.|\,)?)|(and \w+)*)*(\,?\s*\(?\d{4}[a-z]?\)?)\;?)+\)?")
    etAl = re.compile("\(?(\w+ ((et al(\.|\,)?)|(and \w+)*)*(\,?\s*\(?\d{4}[a-z]?\)?)\;?)+\)?")

    # (nye et al, 2018)
    parenthesis =re.compile("(\(\w+ \w*\s*(et al\.?)?\s\d{4}[a-z]?((\,|\;) (\w+ )*\d{4}[a-z]?)*\))")
    # (Küçüktunç et  al. 2012a, b, 2013c, 2015).

    # words that should nor preceed the year
    # TODO: We should consider removing stopwords instead
    predYear=["than","later","since","between","and","during","and","to","the","in","year", "see","january","february","march","april","may","june","july","august","september","october","november","december"]

    keys = []
    filteredList = []
    # print(citation)
    for i in citation:
        i = (x for x in i if x)
        for c in i:
            c = c.replace('-', ',').replace('\\n', '').replace(';', '')
            if (c.__contains__('[')|c.__contains__(']')): # it maches square and square multiple
                CITATION_TYPE = '[NUMBER]'
                nb = c.replace('[', '').replace(']', '').split(',')
                keys.append(nb)
            else:
                CITATION_TYPE = 'AUTHOR_YEAR'
                if(parenthesis.search(c)):
                    grp = parenthesis.search(c).group(0)
                elif (etAl.search(c)):
                    grp = etAl.search(c).group(0)
                else:
                    grp=''
                

                grp = grp.replace(',', '').replace("et al", '').replace("et al.", '').replace('(', '').replace(')', '')
                grp = re.sub(' +', ' ', grp)
                
                list = grp.strip().split(' ')
                if (len(Filter(list, predYear))>1):
                    filteredList = Filter(list, predYear)
                    keys.append(filteredList) if filteredList not in keys else keys

    # print(keys)
    return (keys)

def extractRefs(path):
    references = extract_references_from_file(path)
    extractedReferences = []
    for r in references:
        s =''
        try:
            # print(str(r.keys()))
            s = Reference(''.join(r["raw_ref"]))
            extractedReferences.append(s)
        except KeyError:
            s = Reference(''.join(r))
            extractedReferences.append(s)

    return extractedReferences