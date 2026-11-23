from evals.run_faithfulness_eval import claims, judge_heuristic


def test_claims_keep_citations_with_their_sentence():
    answer = (
        "The launch moved to Oct 29 because of a sync bug [S1]. It hit 8% of accounts [S2, S3].\n\n"
        "- Aisha owns the fix. It ships Oct 20 [S1]\n- An uncited claim about something"
    )
    got = claims(answer)
    assert [ids for _, ids in got] == [[1], [2, 3], [1], []]
    assert got[2][0].startswith("Aisha owns the fix")


def test_heuristic_judge():
    src = ["Aisha: I'll ship the dedupe fix by October 20th and run the backfill on the 21st."]
    assert judge_heuristic("Aisha ships the dedupe fix by October 20th", src) == "supported"
    assert judge_heuristic("Marcus negotiated a lower Plaid price", src) == "unsupported"
