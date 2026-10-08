import pytest
import datetime
from src.query.parser import QueryParser
from src.query.timeparse import parse_time_expression


def test_timeparse_relative():
    now_ref = datetime.datetime(2026, 10, 8, 14, 0, 0)
    
    # 1. in the last hour
    s, e, text = parse_time_expression("did a car pass in the last hour?", now_ref=now_ref)
    assert s == datetime.datetime(2026, 10, 8, 13, 0, 0)
    assert e == now_ref
    
    # 2. past 30 minutes
    s, e, text = parse_time_expression("show pedestrians in the past 30 minutes", now_ref=now_ref)
    assert s == datetime.datetime(2026, 10, 8, 13, 30, 0)
    assert e == now_ref
    
    # 3. between HH:MM and HH:MM
    s, e, text = parse_time_expression("vehicles between 12:00 and 13:00", now_ref=now_ref)
    assert s == datetime.datetime(2026, 10, 8, 12, 0, 0)
    assert e == datetime.datetime(2026, 10, 8, 13, 0, 0)


def test_query_parser_rules():
    parser = QueryParser(known_cameras=["cam_gate", "cam_landscape"])
    now_ref = datetime.datetime(2026, 10, 8, 14, 0, 0)
    
    # Test 1: question with filler words, object, location, and time
    q1 = "did a red car pass near cam_gate in the last hour?"
    p1 = parser.parse(q1, now_ref=now_ref)
    assert "red car" in p1.object_prompt
    assert p1.location == "cam_gate"
    assert p1.t_start is not None
    assert p1.provider == "rules"
    
    # Test 2: pedestrian query on landscape camera
    q2 = "show all pedestrians at cam_landscape"
    p2 = parser.parse(q2, now_ref=now_ref)
    assert "pedestrians" in p2.object_prompt
    assert p2.location == "cam_landscape"
    assert p2.t_start is None  # No time constraint specified
