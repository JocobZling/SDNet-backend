CREATE TABLE `photo` (
    `id` int(11) NOT NULL AUTO_INCREMENT,
    `position` VARCHAR(200),
    `type`VARCHAR(200),
    `userId` int(11),
    `faceScore` VARCHAR(200),
    `colorScore` VARCHAR(200),
    `createTime` TIMESTAMP default current_timestamp,
    PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
