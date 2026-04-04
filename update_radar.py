#!/usr/bin/env python3
"""
Influence Radar 2.0 — 실시간 데이터 갱신 엔진
GitHub Actions에서 매일 한국시간 오전 9시 자동 실행

기능:
1. 인물별 최신 신호 시뮬레이션 (실제 운영 시 NewsAPI/Twitter API 연동)
2. influence_score 기반 좌표 재계산
3. data.json 갱신 및 저장
4. 매수/매도 타이밍 신호 생성
"""

import json
import math
from datetime import datetime, timedelta
import random
import os

class InfluenceRadarUpdater:
    def __init__(self, data_file="data.json"):
        self.data_file = data_file
        self.max_radius = 300
        self.load_data()
    
    def load_data(self):
        """JSON 파일 로드"""
        try:
            with open(self.data_file, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
        except FileNotFoundError:
            print(f"⚠️ {self.data_file} not found. Initialize with empty data.")
            self.data = {"metadata": {}, "influencers": []}
    
    def calculate_coordinates(self, influence_score, angle_offset=0):
        """
        공식: Distance = Max_Radius * (1 - influence_score)
        - influence_score 높을수록 중심(Core)에 가까움
        - 각도는 섹터별로 분배
        
        Returns:
            {"dist": 거리, "angle": 각도(도)}
        """
        dist = self.max_radius * (1 - influence_score)
        return {
            "dist": round(dist, 1),
            "angle": angle_offset
        }
    
    def simulate_news_impact(self, influencer):
        """
        최신 뉴스 시뮬레이션 (실제 API 연동 전 목업)
        실제 운영: NewsAPI/Twitter API 호출 → 감성분석(NLP) → 신호 생성
        
        현재: 무작위 신호 생성 (테스트용)
        """
        sentiments = ["positive", "neutral", "negative"]
        selected_sentiment = random.choice(sentiments)
        
        # 신호 강도 (영향력 점수에 비례)
        signal_strength = influencer["influence_score"] + random.uniform(-0.1, 0.1)
        signal_strength = max(0.0, min(1.0, signal_strength))
        
        return {
            "sentiment": selected_sentiment,
            "strength": round(signal_strength, 2),
            "timestamp": datetime.now().isoformat()
        }
    
    def update_action_plan(self, influencer, signal):
        """
        신호에 따른 매매 액션플랜 동적 조정
        
        Logic:
        - positive → buy_range 하향, sell_target 상향 (매수 기회)
        - negative → buy_range 상향, stop_loss 강화 (리스크 회피)
        - neutral → 유지
        """
        base_buy_range = influencer.get("action_plan", {}).get("buy_range", [100, 120])
        base_sell_target = influencer.get("action_plan", {}).get("sell_target", 150)
        base_stop_loss = influencer.get("action_plan", {}).get("stop_loss", 90)
        
        adjustment = signal["strength"] * 0.05  # 5% 조정폭
        
        if signal["sentiment"] == "positive":
            new_buy_range = [
                round(base_buy_range[0] * (1 - adjustment), 1),
                round(base_buy_range[1] * (1 - adjustment), 1)
            ]
            new_sell_target = round(base_sell_target * (1 + adjustment), 1)
            new_stop_loss = base_stop_loss
        elif signal["sentiment"] == "negative":
            new_buy_range = [
                round(base_buy_range[0] * (1 + adjustment), 1),
                round(base_buy_range[1] * (1 + adjustment), 1)
            ]
            new_sell_target = round(base_sell_target * (1 - adjustment), 1)
            new_stop_loss = round(base_stop_loss * (1 - adjustment), 1)
        else:  # neutral
            new_buy_range = base_buy_range
            new_sell_target = base_sell_target
            new_stop_loss = base_stop_loss
        
        return {
            "buy_range": new_buy_range,
            "sell_target": new_sell_target,
            "stop_loss": new_stop_loss
        }
    
    def update_all(self):
        """
        모든 influencer의 데이터 갱신 및 신호 생성
        """
        print("🔄 Updating Influence Radar data...")
        
        for influencer in self.data.get("influencers", []):
            # 1. 뉴스 시뮬레이션
            signal = self.simulate_news_impact(influencer)
            
            # 2. 좌표 재계산
            angle = influencer["coordinates"].get("angle", 0)
            influencer["coordinates"] = self.calculate_coordinates(
                influencer["influence_score"], 
                angle
            )
            
            # 3. 액션플랜 동적 조정
            influencer["action_plan"] = self.update_action_plan(influencer, signal)
            
            # 4. 신호 기록
            influencer["last_signal"] = datetime.now().strftime("%Y-%m-%d")
            influencer["signal_sentiment"] = signal["sentiment"]
            
            print(f"✅ {influencer['name']}: {signal['sentiment'].upper()} "
                  f"(strength: {signal['strength']})")
        
        # 5. 메타데이터 갱신
        self.data["metadata"]["last_updated"] = datetime.now().isoformat()
        
        print(f"✨ Update complete. Total influencers: {len(self.data['influencers'])}")
    
    def save_data(self):
        """갱신된 JSON 저장"""
        with open(self.data_file, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
        print(f"💾 Data saved to {self.data_file}")
    
    def generate_trading_signals(self):
        """
        종목별 매매 신호 생성
        수익 모델 검증: 신호 신뢰도 vs 실제 수익률
        """
        signals_output = []
        for influencer in self.data.get("influencers", []):
            for ticker in influencer.get("ticker", []):
                signal_record = {
                    "ticker": ticker,
                    "source": influencer["name"],
                    "signal": influencer["signal_sentiment"],
                    "buy_range": influencer["action_plan"]["buy_range"],
                    "sell_target": influencer["action_plan"]["sell_target"],
                    "stop_loss": influencer["action_plan"]["stop_loss"],
                    "timestamp": datetime.now().isoformat()
                }
                signals_output.append(signal_record)
        
        return signals_output

def main():
    updater = InfluenceRadarUpdater("data.json")
    updater.update_all()
    updater.save_data()
    
    # 매매 신호 출력
    signals = updater.generate_trading_signals()
    print("\n📊 Trading Signals Generated:")
    for sig in signals:
        print(f"  {sig['ticker']} ({sig['source']}): {sig['signal'].upper()}")

if __name__ == "__main__":
    main()
