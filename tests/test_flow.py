"""Layer 2: Flow 定义测试"""
import sys
import os
import pytest
from pocketflow import Flow

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Server_Ask"))

from rag_logic.flow import offline_flow, online_flow


class TestOfflineFlow:
    """离线 Flow 测试"""

    def test_flow_exists(self):
        """offline_flow 存在"""
        assert offline_flow is not None

    def test_flow_has_start_node(self):
        """offline_flow 有起始节点"""
        assert hasattr(offline_flow, "start")

    def test_flow_is_flow_instance(self):
        """offline_flow 是 Flow 实例"""
        assert isinstance(offline_flow, Flow)


class TestOnlineFlow:
    """在线 Flow 测试"""

    def test_flow_exists(self):
        """online_flow 存在"""
        assert online_flow is not None

    def test_flow_has_start_node(self):
        """online_flow 有起始节点"""
        assert hasattr(online_flow, "start")

    def test_flow_is_flow_instance(self):
        """online_flow 是 Flow 实例"""
        assert isinstance(online_flow, Flow)
