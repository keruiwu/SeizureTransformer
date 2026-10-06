import numpy as np
import pytest

from lteeg.data.annotations import (IGNORE_INDEX, AnnotationError, Event, events_to_intervals,
                                    intervals_to_labels, parse_annotation_file, positive_counts, validate_events)


def write(tmp_path, text, name="p_annotations.txt", bom=False):
    p = tmp_path / name
    p.write_bytes((b"\xef\xbb\xbf" if bom else b"") + text.encode("utf-8"))
    return p


def test_parse_sections_header_bom_crlf(tmp_path):
    text = ("# comment\r\n[files]\r\nfile\tstart\r\nchb01_01.h5\t0\r\n\r\n[seizures]\r\n"
            "file\tstart_sec\tend_sec\r\nchb01_03.h5\t2996\t3036\r\nchb01_04.h5\t1467.5\t1494\r\n[other]\r\nx\t1\t2\r\n")
    rows = parse_annotation_file(write(tmp_path, text, bom=True))
    assert [(r.file, r.event.start, r.event.end) for r in rows] == [
        ("chb01_03.h5", 2996.0, 3036.0), ("chb01_04.h5", 1467.5, 1494.0)]
    assert rows[0].line == 8


def test_parse_errors_name_the_line(tmp_path):
    with pytest.raises(AnnotationError, match=r":3: start/end are not numbers"):
        parse_annotation_file(write(tmp_path, "[seizures]\nfile\tstart\tend\nchb01_03.h5\tabc\t3036\n"))
    with pytest.raises(AnnotationError, match="not found"):
        parse_annotation_file(write(tmp_path, "[other]\nx\t1\t2\n"))
    with pytest.raises(AnnotationError, match="appears twice"):
        parse_annotation_file(write(tmp_path, "[seizures]\na\t1\t2\n[seizures]\nb\t1\t2\n"))
    with pytest.raises(AnnotationError, match="expected"):
        parse_annotation_file(write(tmp_path, "[seizures]\nchb01_03.h5\t12\n"))


def test_label_column(tmp_path):
    p = write(tmp_path, "[seizures]\nfile\tstart\tend\ttype\na.h5\t1\t2\tfocal\nb.h5\t3\t4\tgen\n")
    rows = parse_annotation_file(p, label_column=3, label_map={"focal": 1, "gen": 2})
    assert [r.event.label for r in rows] == [1, 2]
    with pytest.raises(AnnotationError, match="not in data.label_map"):
        parse_annotation_file(p, label_column=3, label_map={"focal": 1})


def test_validate_events():
    evs, warns = validate_events([Event(10, 20), Event(0, 5)], 100.0, "r")
    assert [e.start for e in evs] == [0, 10] and not warns
    evs, warns = validate_events([Event(90, 100.5)], 100.0, "r", end_tolerance_sec=1.0)
    assert evs[0].end == 100.0 and warns
    with pytest.raises(AnnotationError, match="beyond recording end"):
        validate_events([Event(90, 102)], 100.0, "r", end_tolerance_sec=1.0)
    with pytest.raises(AnnotationError, match="overlapping"):
        validate_events([Event(0, 10), Event(5, 15)], 100.0, "r")
    evs, _ = validate_events([Event(0, 10), Event(5, 15)], 100.0, "r", overlapping="merge")
    assert evs == [Event(0, 15)]
    with pytest.raises(AnnotationError, match="invalid event"):
        validate_events([Event(5, 5)], 100.0, "r")


def test_intervals_and_labels():
    iv = events_to_intervals([Event(1.0, 2.0), Event(3.5, 4.0, 2)], 4.0, 20)
    assert iv.tolist() == [[4, 8, 1], [14, 16, 2]]
    y = intervals_to_labels(iv, 2, 16)
    assert y.tolist() == [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0]
    y = intervals_to_labels(iv, 0, 20, ignore_margin=1)
    # margin of 1 sample around onset (4) and offset (8): [3, 5) and [7, 9) are ignored
    assert y[2:10].tolist() == [0, IGNORE_INDEX, IGNORE_INDEX, 1, 1, IGNORE_INDEX, IGNORE_INDEX, 0]


def test_positive_counts_matches_bruteforce():
    rng = np.random.default_rng(0)
    n, w = 5000, 300
    iv = np.array([[100, 450, 1], [1200, 1210, 1], [2000, 3000, 1]])
    starts = rng.integers(0, n - w, size=200)
    mask = np.zeros(n, bool)
    for a, b, _ in iv:
        mask[a:b] = True
    expected = np.array([mask[s:s + w].sum() for s in starts])
    assert np.array_equal(positive_counts(iv, starts, w), expected)
