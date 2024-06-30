import spacy

nlp = spacy.load("en_core_web_sm")


def check_compliance(text):
    doc = nlp(text)
    compliance_issues = [ent.text for ent in doc.ents if ent.label_ == "LAW"]
    return compliance_issues
