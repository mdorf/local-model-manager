# tests/test_logtail.py
from lmm.logtail import read_log_tail, tail_new_lines


def test_read_log_tail_returns_last_lines(tmp_path):
    p = tmp_path / "s.log"
    p.write_text("".join(f"line{i}\n" for i in range(10)))
    assert read_log_tail(p, max_lines=3) == ["line7", "line8", "line9"]


def test_read_log_tail_missing_file(tmp_path):
    assert read_log_tail(tmp_path / "nope.log") == []


def test_tail_new_lines_advances_offset(tmp_path):
    p = tmp_path / "s.log"
    p.write_text("a\nb\n")
    lines, off = tail_new_lines(p, 0)
    assert lines == ["a", "b"] and off == 4
    p.write_text("a\nb\nc\n")
    lines2, off2 = tail_new_lines(p, off)
    assert lines2 == ["c"] and off2 == 6
