package cn.sd.entities.vo;

import lombok.*;

import java.util.List;

@Setter
@Getter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class FacePositionAndNameAndList {
    String facePosition;
    String clusteringName;
    List<PhotoDisplayVo> photoDisplayVoList;
}
