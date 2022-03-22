package cn.sd.entities.vo;

import lombok.*;

@Setter
@Getter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class ImageDisplayVo {
    private String pictureOnePosition;
    private String pictureTwoPosition;
    private String originalImagePosition;
    private Long detectionId;
}
