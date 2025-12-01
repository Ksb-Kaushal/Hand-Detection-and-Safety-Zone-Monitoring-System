# Hand-Detection-and-Safety-Zone-Monitoring-System
Real Time Industrial Safety Monitoring with Hikvision RTSP Camera, MediaPipe, YOLOv8 and Automated PLC Triggering
# 1. Problem Statement

In **Bridgestone** rubber processing facility, operators work near a high-RPM rubber-flattening roller machine.
If a worker’s hand enters a critical area, the machine can pull the arm into the moving roller, causing severe injury or fatal accidents.

Traditional safety guards did not fully prevent unsafe proximity. The requirement was to design a vision-based safety solution that can:

*Detect human hand(s) near the roller

*Identify unsafe proximity zones

*Trigger warning when a hand enters a caution area

*Trigger machine stop when a hand enters a danger area

*Record evidence video during emergency events

*Output 24V DC signal to PLC / relay

This repository contains the prototype implementation of that system


