#!/bin/bash
# The source code is taken from https://github.com/emp-toolkit/emp-readme/blob/master/scripts/throttle.sh
DEV=lo 
if [ "$1" == "del" ]
then
	sudo tc qdisc del dev $DEV root
fi

# This LAN and WAN network configuration based on the setting in Cryptflow2 and Cheetah, 
# which is available from https://github.com/Alibaba-Gemini-Lab/OpenCheetah/tree/main/scripts.
if [ "$1" == "lan" ]
then
sudo tc qdisc del dev $DEV root
sudo tc qdisc add dev $DEV root handle 1: tbf rate 3000mbit burst 100000 limit 10000
sudo tc qdisc add dev $DEV parent 1:1 handle 10: netem delay 0.15msec
fi
if [ "$1" == "wan" ]
then
sudo tc qdisc del dev $DEV root
sudo tc qdisc add dev $DEV root handle 1: tbf rate 400mbit burst 100000 limit 10000
sudo tc qdisc add dev $DEV parent 1:1 handle 10: netem delay 20msec
fi
# This mobile_US_avg network configuration based on the real-world network performance data reported on Ookla dataset, 
# which is available from https://www.speedtest.net/global-index/united-states.
if [ "$1" == "mobile_US_avg" ]
then
sudo tc qdisc del dev $DEV root
sudo tc qdisc add dev $DEV root handle 1: tbf rate 125mbit burst 100000 limit 10000
sudo tc qdisc add dev $DEV parent 1:1 handle 10: netem delay 15msec
fi
