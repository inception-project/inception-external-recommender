from ariadne.contrib.inception_util import *

from cassis import Cas, TypeSystem


def test_create_prediction():
    typesystem = TypeSystem()
    Span = typesystem.create_type("custom.Span")
    typesystem.add_feature(Span, "inception_internal_predicted", "uima.cas.Boolean")
    typesystem.add_feature(Span, "value", "uima.cas.String")
    typesystem.add_feature(Span, "value_score", "uima.cas.Double")
    typesystem.add_feature(Span, "value_score_explanation", "uima.cas.String")
    typesystem.add_feature(Span, "value_auto_accept", "uima.cas.Boolean")
    cas = Cas(typesystem=typesystem)
    prediction = create_prediction(
        cas, "custom.Span", "value", 0, 4, "label", score=0.1, score_explanation="blah", auto_accept=True
    )
    assert prediction.get("begin") == 0
    assert prediction.get("end") == 4
    assert prediction.get("value") == "label"
    assert prediction.get("inception_internal_predicted") == True
    assert prediction.get("value_score") == 0.1
    assert prediction.get("value_score_explanation") == "blah"
    assert prediction.get("value_auto_accept") == True
