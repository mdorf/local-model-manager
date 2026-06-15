import threading

from lmm.state import InstanceRecord, load_instances, mutate_instances


def test_mutate_instances_adds_under_lock(monkeypatch, tmp_path):
    monkeypatch.setenv("LMM_STATE_DIR", str(tmp_path / "st"))

    def add(recs):
        recs.append(InstanceRecord(port=8080, pid=1, model_path="/m/a.gguf",
                                   started_at=1.0))
        return recs

    out = mutate_instances(add)
    assert [r.port for r in out] == [8080]
    assert [r.port for r in load_instances()] == [8080]


def test_concurrent_mutations_lose_no_records(monkeypatch, tmp_path):
    monkeypatch.setenv("LMM_STATE_DIR", str(tmp_path / "st"))
    n = 40

    def make(i):
        def add(recs):
            recs.append(InstanceRecord(port=9000 + i, pid=i, model_path=f"/m/{i}",
                                       started_at=float(i)))
            return recs
        return add

    threads = [threading.Thread(target=lambda i=i: mutate_instances(make(i)))
               for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    ports = sorted(r.port for r in load_instances())
    assert ports == [9000 + i for i in range(n)]
