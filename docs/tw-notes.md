label studio template for keypoints + bounding box:

```
  <View>
    <RectangleLabels name="label" toName="img-1">
   	<Label value="scale" background="green"/>
    </RectangleLabels>
    <KeyPointLabels name="kp-1" toName="img-1">
      
      
    <Label value="tick_1" background="#FFA39E"/><Label value="tick_2" background="#D4380D"/><Label value="tick_3" background="#FFC069"/><Label value="tick_4" background="#AD8B00"/><Label value="tick_5" background="#D3F261"/></KeyPointLabels>
    <Image name="img-1" value="$img"/>
  </View>
```
