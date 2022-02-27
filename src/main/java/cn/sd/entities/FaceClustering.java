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
public class FaceClustering {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;
    @JsonProperty("photoId")
    private Long photoId;
    @JsonProperty("position")
    private String position;
    @JsonProperty("clusteringId")
    private Long clusteringId;
    @JsonProperty("userId")
    private Long userId;
    @JsonProperty("airFaceId")
    private String airFaceId;
    @JsonFormat(pattern = "yyyy-MM-dd", timezone = "GMT+8")
    private Date createTime = new Date();
}
