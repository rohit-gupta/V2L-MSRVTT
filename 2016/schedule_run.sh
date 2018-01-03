until [[ "$(date)" =~ "04:10:" ]]; do
    sleep 10
done
UDA_VISIBLE_DEVICES=0 python test_tag_model.py -t action_tags_med -l 256
