import { View, Text, TouchableOpacity, Image } from "react-native";

import styles from "./popularjobcard.style";
import { checkImageURL } from "../../../../utils";
import { CircularProgressBar } from "../../../../components";

const PopularJobCard = ({ book, selectedBook, handleCardPress }) => {
  return (
    <TouchableOpacity
      style={styles.container(selectedBook, book)}
      onPress={() => handleCardPress(book)}
    >
      <View style={styles.infoHeader}>
      <TouchableOpacity style={styles.logoContainer(selectedBook, book)}>
        <Image
          source={{
            uri: checkImageURL(book.thumbnail)
              ? book.thumbnail
              : "https://t4.ftcdn.net/jpg/05/05/61/73/360_F_505617309_NN1CW7diNmGXJfMicpY9eXHKV4sqzO5H.jpg",
          }}
          resizeMode='contain'
          style={styles.logoImage}
        />
      </TouchableOpacity>
      <TouchableOpacity style={styles.scoreContainer(selectedBook, book)}>
      {/* <Text>
        {Math.ceil(book.score[0])}
      </Text> */}
      <CircularProgressBar 
        percentage={Math.ceil(book.score[0])}
      />
      </TouchableOpacity>
      </View>

      <Text style={styles.companyName} numberOfLines={1}>
        {book.authors[0]}
      </Text>

      <View style={styles.infoContainer}>
        <Text style={styles.jobName(selectedBook, book)} numberOfLines={2}>
          {book.title}
        </Text>
        <View style={styles.infoWrapper}>
          <Text style={styles.publisher(selectedBook, book)} numberOfLines={2}>
            {book.subtitle}
          </Text>
        </View>
      </View>
    </TouchableOpacity>
  );
};

export default PopularJobCard;
