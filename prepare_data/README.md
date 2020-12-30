# download data
```bash
wget --header="Host: dl.challenge.zalo.ai" --header="User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.111 Safari/537.36" --header="Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9" --header="Accept-Language: en-US,en;q=0.9,vi-VN;q=0.8,vi;q=0.7" --header="Referer: https://challenge.zalo.ai/" --header="Cookie: _ga=GA1.2.1869849351.1604392068; _gid=GA1.2.2084818595.1604392068; __zi=3000.SSZzejyD0jydXQcYsa00d3xBfxgP71AM8Tdbg8y9KSbdtERbmGzGp6FRgU7211d5RzAaxCmD6Sa.1; zaiac_c=6elQ0Vlhi3COWzu6_UIqE1w3t0_YkDHWMe-x3E6HWtnNgECSnB_CSYMgnmN2YyXpPAd64OMMvYKviOmHejBdOZZt_pEIvV1M8lF08Rdyj5SPz88nai-KP0Bd-G2Qz90m9_EAGy2MaNLsygCTmiUGLLJF-XRawCbJRDhaOyh0Y2LjvAPhpVlELIdUqYEL_-Pl9js899RUg4ycz9G1ZkZvDpdK_qUD_Tuc1fZp5-Qb-NHJakW2j97BI0gatGwUmlSu2yZlSP-XtoOBXCnjkhYa05FIK5a7-_MoDG; fpsend=148555; _gat_gtag_UA_108352130_23=1" --header="Connection: keep-alive" "https://dl.challenge.zalo.ai/traffic-sign-detection/data/za_traffic_2020.zip" -c -O 'za_traffic_2020.zip'

unzip za_traffic_2020.zip
```

# merge overlay bbox
```bash
python3 merge_over_lay_bbox.py
```
# k fold
```bash
python3 k_fold.py
```

# split big image to small image
```bash
python3 split_image.py
```
# create data fold
```bash
python3 create_data_fold.py --fold <1 to 5>
python3 create_label_fold.py --fold <1 to 5>
```

