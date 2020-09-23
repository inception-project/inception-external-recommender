from pathlib import Path

from cassis import *

from ariadne.contrib.jieba import JiebaSegmenter

_PREDICTED_TYPE = "ariadne.test.jieba"
_PREDICTED_FEATURE = "value"


def test_predict(tmpdir_factory):
    model_directory = Path(tmpdir_factory.mktemp("models"))
    sut = JiebaSegmenter(model_directory)

    cas = _load_data()

    sut.predict(cas, _PREDICTED_TYPE, _PREDICTED_FEATURE, "test_project", "doc_42", "test_user")

    for prediction in cas.select(_PREDICTED_TYPE):
        assert prediction.value is not None


def _load_data() -> Cas:
    text = """
    一对年轻的夫妇对面搬来一户新邻居。第二天早上，当他们吃早饭的时候，年轻的妻子看到了新搬来的邻居正在外面洗衣服。

    妻子对丈夫说道：“那些衣服洗得不干净，也许那个邻居不知道如何清洗。也许她需要好一点的洗衣粉。”
    
    丈夫看了看了妻子，沉默不语。
    
    就这样每次邻居洗衣服，妻子都会这样评论对方一番。 大概一个月后，年轻的妻子惊奇地发现，邻居的晾衣绳上居然悬挂着一件干净的衣服，她大叫着对丈夫说：“快看！她学会洗衣服了。我想知道是谁教会她这个的呢？”
    
    她的丈夫却回答到：“我今天早上一大早起来，然后我把玻璃悬擦干净了。”
    
    在我们作出判断之前，首先要看一下你的“窗户”是否干净。
    """
    cas = Cas()
    cas.sofa_string = text.strip()
    predicted_type = cas.typesystem.create_type(_PREDICTED_TYPE)
    cas.typesystem.add_feature(predicted_type, _PREDICTED_FEATURE, "uima.cas.String")
    cas.typesystem.add_feature(predicted_type, "inception_internal_predicted", "uima.cas.Boolean")

    return cas
