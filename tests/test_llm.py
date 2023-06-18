import os
from pytest import fixture


@fixture()
def txt():
    t = """After his release from the Israel Defense Forces in 1990, Hess worked as a copywriter in an advertising 
    agency and later as a journalist in the Shishi (Friday) newspaper. During his studies in the Hebrew University, 
    he joined the Shorashim Institute for Jewish Studies. Hess conducted and directed seminars at the Institute until 
    he was appointed vice CEO of the Institute in 1996, a position he held until 1999. In 2000 Hess started serving 
    as the Jewish Agency's shaliach to Tucson, Arizona.[3] He wrote a weekly column in the Arizona Jewish Post and 
    won an excellence Award (2002) from the Association of Jewish Centers in North America (now known as the JCCA, 
    the JCC Association) for the many cultural programs he initiated in Arizona and an Excellence Award on behalf of 
    the Jewish Federation (2003). He was also a regular commentator about Israel on the weekly program The Too Jewish 
    Radio Show with Rabbi Sam Cohon and Friends during his tenure in Tucson. He returned to Israel in 2003 and 
    continued working in the Jewish Agency as the director of partnerships between Jewish communities abroad and in 
    Israel."""
    return t


@fixture()
def api_key():
    return os.environ.get("OPENAI_API_KEY", "")


def test_index(txt, api_key):
    from slse import SLSE
    dir_path = os.path.dirname(os.path.realpath(__file__))
    slse = SLSE(dir_path=dir_path, api_key=api_key)
    texts = slse.chunk_text(txt, 1000)
    ix = slse.get_index("test", texts)
    res = slse.query("Who is Hess?", index=ix)
    print(res)


def test_refine_llm(txt, api_key):
    from slse import SLSE
    dir_path = os.path.dirname(os.path.realpath(__file__))
    slse = SLSE(dir_path=dir_path, api_key=api_key)
    texts = slse.chunk_text(txt, 1000)
    ix = slse.get_index_define_llm("test2", texts)
    res = slse.query("Who is Hess?", index=ix)
    print(res)


def test_refine_llm_load(api_key):
    from slse import SLSE
    dir_path = os.path.dirname(os.path.realpath(__file__))
    slse = SLSE(dir_path=dir_path, api_key=api_key)
    # texts = slse.chunk_text(txt, 1000)
    ix = slse.get_index_define_llm("test2")
    res = slse.query("Who is Hess?", index=ix)
    print(res)
