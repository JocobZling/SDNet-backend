package cn.sd.repositories;


import cn.sd.entities.NewDetection;
import org.springframework.data.jpa.repository.JpaRepository;

public interface DetectionRepository extends JpaRepository<NewDetection, Long> {
}
