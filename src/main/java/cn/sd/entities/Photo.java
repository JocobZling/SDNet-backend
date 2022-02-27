package cn.sd.entities;

import com.fasterxml.jackson.annotation.JsonFormat;
import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.*;

import javax.persistence.*;
import java.util.Date;

@Setter
@Getter
@NoArgsConstructor
@AllArgsConstructor
@Builder
@Entity

public class Photo{
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;
    @JsonProperty("position")
    private String position;
    @JsonProperty("type")
    private String type;
    @JsonProperty("userId")
    private Long userId;
    @JsonProperty("faceScore")
    private String faceScore;
    @JsonProperty("colorScore")
    private String colorScore;
    @JsonFormat(pattern = "yyyy-MM-dd", timezone = "GMT+8")
    private Date createTime = new Date();

}
