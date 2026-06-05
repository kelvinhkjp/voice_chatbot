プロジェクト概要
Ambient Voice Assistant は、複数話者対応のリアルタイム音声チャットボットです。ブラウザのマイクから音声を録音し、文字起こし・話者分離・AI応答までを自動で処理します。

使用技術
バックエンド

FastAPI — 高速なPython製WebAPIフレームワーク
faster-whisper — OpenAI Whisperの最適化版。ローカルで音声をテキストに変換（文字起こし）
pyannote.audio 4.x — 話者分離（誰が話したかを識別）
LangChain + Gemini 2.5 Flash — Google製大規模言語モデルによるAI応答生成
FFmpeg — 音声フォーマット変換（WebM → WAV）
soundfile / torch — 音声データのメモリ内処理（torchcodec DLL問題を回避）

フロントエンド

HTML / CSS / JavaScript — シングルページのチャットUI
MediaRecorder API — ブラウザからの音声録音
Web Speech Synthesis API — AIの返答をブラウザで音声再生（TTS）
ノイズキャンセリング — echoCancellation / noiseSuppression / autoGainControl をブラウザAPIで有効化


主な特徴

ブラウザ内ノイズキャンセリング対応
複数話者の識別と表示
会話履歴を保持したAI応答（最大20ターン）
セッション管理とリセット機能
CPUのみでも動作可能（GPU対応も可）
