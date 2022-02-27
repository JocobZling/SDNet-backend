CREATE TABLE `clustering` (
    `id` int(11) NOT NULL AUTO_INCREMENT,
    `clusterName` VARCHAR(200),
    `userId` int(11),
    `createTime` TIMESTAMP default current_timestamp,
    PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
