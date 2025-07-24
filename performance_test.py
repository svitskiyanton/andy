#!/usr/bin/env python3
"""
Performance test to compare single vs parallel connection overhead
"""

import asyncio
import time
import psutil
import os
from typing import List

class PerformanceMonitor:
    def __init__(self):
        self.start_time = time.time()
        self.start_memory = psutil.Process().memory_info().rss
        self.events = []
    
    def mark(self, event_name: str):
        current_time = time.time() - self.start_time
        current_memory = psutil.Process().memory_info().rss
        self.events.append({
            'time': current_time,
            'memory': current_memory,
            'event': event_name
        })
        print(f"⏱️  [{current_time:.3f}s] {event_name} - Memory: {current_memory / 1024 / 1024:.1f}MB")
    
    def summary(self):
        print("\n" + "="*60)
        print("📊 PERFORMANCE SUMMARY")
        print("="*60)
        
        for i, event in enumerate(self.events):
            if i > 0:
                prev_time = self.events[i-1]['time']
                prev_memory = self.events[i-1]['memory']
                time_diff = (event['time'] - prev_time) * 1000
                memory_diff = (event['memory'] - prev_memory) / 1024 / 1024
                print(f"⏱️  [{event['time']:.3f}s] {event['event']} (+{time_diff:.1f}ms, +{memory_diff:.1f}MB)")
            else:
                print(f"⏱️  [{event['time']:.3f}s] {event['event']} - Memory: {event['memory'] / 1024 / 1024:.1f}MB")
        
        total_time = self.events[-1]['time'] if self.events else 0
        total_memory = (self.events[-1]['memory'] - self.start_memory) / 1024 / 1024 if self.events else 0
        print(f"\n📈 Total: {total_time:.3f}s, Memory: +{total_memory:.1f}MB")
        print("="*60)

async def simulate_single_connection():
    """Simulate single connection overhead"""
    monitor = PerformanceMonitor()
    
    monitor.mark("Single connection start")
    
    # Simulate WebSocket connection
    await asyncio.sleep(0.1)
    monitor.mark("WebSocket connected")
    
    # Simulate audio processing
    for i in range(10):
        await asyncio.sleep(0.05)
        monitor.mark(f"Audio chunk {i+1}")
    
    # Simulate transcript processing
    await asyncio.sleep(0.1)
    monitor.mark("Transcript processed")
    
    return monitor

async def simulate_parallel_connection():
    """Simulate parallel connection overhead"""
    monitor = PerformanceMonitor()
    
    monitor.mark("Parallel connection start")
    
    # Simulate connection manager
    await asyncio.sleep(0.05)
    monitor.mark("Connection manager created")
    
    # Simulate active connection
    await asyncio.sleep(0.1)
    monitor.mark("Active connection established")
    
    # Simulate backup connection
    await asyncio.sleep(0.1)
    monitor.mark("Backup connection established")
    
    # Simulate audio processing through layers
    for i in range(10):
        await asyncio.sleep(0.05)
        monitor.mark(f"Audio chunk {i+1} (through layers)")
    
    # Simulate transcript processing through layers
    await asyncio.sleep(0.15)
    monitor.mark("Transcript processed (through layers)")
    
    # Simulate TTS processing
    await asyncio.sleep(0.1)
    monitor.mark("TTS processed")
    
    return monitor

async def main():
    print("🚀 Performance Test: Single vs Parallel Connection")
    print("="*60)
    
    print("\n📊 Testing Single Connection...")
    single_monitor = await simulate_single_connection()
    single_monitor.summary()
    
    print("\n📊 Testing Parallel Connection...")
    parallel_monitor = await simulate_parallel_connection()
    parallel_monitor.summary()
    
    # Compare results
    single_time = single_monitor.events[-1]['time'] if single_monitor.events else 0
    parallel_time = parallel_monitor.events[-1]['time'] if parallel_monitor.events else 0
    single_memory = (single_monitor.events[-1]['memory'] - single_monitor.start_memory) / 1024 / 1024 if single_monitor.events else 0
    parallel_memory = (parallel_monitor.events[-1]['memory'] - parallel_monitor.start_memory) / 1024 / 1024 if parallel_monitor.events else 0
    
    print("\n📈 COMPARISON:")
    print(f"⏱️  Time: Single {single_time:.3f}s vs Parallel {parallel_time:.3f}s")
    print(f"📊 Memory: Single +{single_memory:.1f}MB vs Parallel +{parallel_memory:.1f}MB")
    print(f"🚀 Speed: Parallel is {parallel_time/single_time:.1f}x slower")
    print(f"💾 Memory: Parallel uses {parallel_memory/single_memory:.1f}x more memory")

if __name__ == "__main__":
    asyncio.run(main()) 