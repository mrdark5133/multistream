import pytest
import datetime
from src.query.parser import QueryParser
from src.query.timeparse import parse_time_expression


def test_timeparse_8_cases():
    """Verify at least 8 time parsing cases with 'now' as reference timestamp."""
    now_ref = datetime.datetime(2026, 10, 8, 19, 0, 0)
    
    # Case 1: last hour
    s1, e1, span1 = parse_time_expression("a red car in the last hour", now_ref=now_ref)
    assert s1 == datetime.datetime(2026, 10, 8, 18, 0, 0)
    assert e1 == now_ref
    assert "last hour" in span1
    
    # Case 2: yesterday
    s2, e2, span2 = parse_time_expression("a delivery van yesterday", now_ref=now_ref)
    assert s2 == datetime.datetime(2026, 10, 7, 0, 0, 0)
    assert e2.date() == datetime.date(2026, 10, 7)
    assert "yesterday" in span2
    
    # Case 3: between 2 and 3 PM
    s3, e3, span3 = parse_time_expression("someone walking between 2 and 3 PM", now_ref=now_ref)
    assert s3 == datetime.datetime(2026, 10, 8, 14, 0, 0)
    assert e3 == datetime.datetime(2026, 10, 8, 15, 0, 0)
    assert "between 2 and 3 pm" in span3.lower()
    
    # Case 4: this morning
    s4, e4, span4 = parse_time_expression("a bicyclist this morning", now_ref=now_ref)
    assert s4 == datetime.datetime(2026, 10, 8, 6, 0, 0)
    assert e4 == datetime.datetime(2026, 10, 8, 12, 0, 0)
    assert "this morning" in span4
    
    # Case 5: this afternoon
    s5, e5, span5 = parse_time_expression("a courier this afternoon", now_ref=now_ref)
    assert s5 == datetime.datetime(2026, 10, 8, 12, 0, 0)
    assert e5 == datetime.datetime(2026, 10, 8, 18, 0, 0)
    assert "this afternoon" in span5
    
    # Case 6: past 30 minutes
    s6, e6, span6 = parse_time_expression("a car in the past 30 minutes", now_ref=now_ref)
    assert s6 == datetime.datetime(2026, 10, 8, 18, 30, 0)
    assert e6 == now_ref
    assert "past 30 minutes" in span6
    
    # Case 7: today
    s7, e7, span7 = parse_time_expression("any intruder today", now_ref=now_ref)
    assert s7 == datetime.datetime(2026, 10, 8, 0, 0, 0)
    assert e7 == now_ref
    assert "today" in span7
    
    # Case 8: in the last 2 hours
    s8, e8, span8 = parse_time_expression("vehicles in the last 2 hours", now_ref=now_ref)
    assert s8 == datetime.datetime(2026, 10, 8, 17, 0, 0)
    assert e8 == now_ref
    assert "last 2 hours" in span8


def test_query_parser_rules_and_unresolved_location():
    # Known cameras in system
    parser = QueryParser(
        known_cameras=["cam_landscape", "cam_landscape2"],
        known_aliases={"back alley": "cam_landscape2"}
    )
    now_ref = datetime.datetime(2026, 10, 8, 19, 0, 0)
    
    # Test resolved camera
    p_res = parser.parse("a red car near cam_landscape in the last hour", now_ref=now_ref)
    assert p_res.location == "cam_landscape"
    assert p_res.location_status == "RESOLVED"
    assert p_res.resolved_camera == "cam_landscape"
    assert "red car" in p_res.object_prompt
    
    # Test resolved alias
    p_alias = parser.parse("a dog in the back alley", now_ref=now_ref)
    assert p_alias.location == "back alley"
    assert p_alias.location_status == "RESOLVED"
    assert p_alias.resolved_camera == "cam_landscape2"
    
    # Test UNRESOLVED location: "a person at the main gate"
    p_unres = parser.parse("a person at the main gate", now_ref=now_ref)
    assert p_unres.location == "main gate"
    assert p_unres.location_status == "UNRESOLVED"
    assert p_unres.resolved_camera is None
    assert p_unres.object_prompt == "a person"
