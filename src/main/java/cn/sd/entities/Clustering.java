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
public class Clustering {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;
    @JsonProperty("clusterName")
    private String clusterName;
    @JsonProperty("userId")
    private Long userId;
    @JsonFormat(pattern = "yyyy-MM-dd", timezone = "GMT+8")
    @Builder.Default
    private Date createTime = new Date();
}
