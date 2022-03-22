package cn.sd.entities.vo;

import lombok.*;

@Setter
@Getter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class AnalysisVo {
    private String pictureOriginal;
    private String pictureOne;
    private String pictureTwo;
    private String histOne;
    private String histTwo;
    private String histOriginal;
}
