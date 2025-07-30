import React from 'react';
import {
  View,
  Text,
  StyleSheet,
  SafeAreaView,
  TextInput,
  TouchableOpacity,
  FlatList,
  Image,
} from 'react-native';
import { NavigationContainer } from '@react-navigation/native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import Icon from 'react-native-vector-icons/Ionicons';

const Tab = createBottomTabNavigator();

const dummyCards = [
  {
    id: '1',
    name: 'Charizard EX',
    image: require('./assets/sv3-5_en_199_std.jpg'),
  },
  {
    id: '2',
    name: 'Pikachu V',
    image: require('./assets/sv3-5_en_199_std.jpg'),
  },
];

const SearchBar = () => (
  <View style={styles.searchContainer}>
    <TextInput
      placeholder="Search Pokémon cards"
      placeholderTextColor="#aaa"
      style={styles.searchInput}
    />
    <TouchableOpacity style={styles.cameraIcon}>
      <Icon name="camera-outline" size={22} color="#fff" />
    </TouchableOpacity>
  </View>
);

const CardList = () => (
  <FlatList
    data={dummyCards}
    horizontal
    keyExtractor={(item) => item.id}
    showsHorizontalScrollIndicator={false}
    contentContainerStyle={{ paddingHorizontal: 10 }}
    renderItem={({ item }) => (
      <View style={styles.card}>
        <Image source={item.image} style={styles.cardImage} />
        <Text style={styles.cardTitle}>{item.name}</Text>
      </View>
    )}
  />
);

// --- Screens ---
const HomeScreen = () => (
  <SafeAreaView style={styles.screen}>
    <SearchBar />
    <Text style={styles.sectionTitle}>Featured Cards</Text>
    <CardList />
  </SafeAreaView>
);

const PopularScreen = () => (
  <SafeAreaView style={styles.centered}><Text>Popular Cards</Text></SafeAreaView>
);

const ScanScreen = () => (
  <SafeAreaView style={styles.centered}><Text>Scan a Card</Text></SafeAreaView>
);

const CollectionScreen = () => (
  <SafeAreaView style={styles.centered}><Text>My Collection</Text></SafeAreaView>
);

const ProfileScreen = () => (
  <SafeAreaView style={styles.centered}><Text>My Profile</Text></SafeAreaView>
);

export default function App() {
  return (
    <NavigationContainer>
      <Tab.Navigator
        screenOptions={({ route }) => ({
          headerShown: false,
          tabBarStyle: styles.tabBar,
          tabBarShowLabel: false,
          tabBarIcon: ({ color, size }) => {
            let iconName = 'home-outline';
            switch (route.name) {
              case 'Home': iconName = 'home-outline'; break;
              case 'Popular': iconName = 'flame-outline'; break;
              case 'Scan': iconName = 'scan-outline'; break;
              case 'Collection': iconName = 'albums-outline'; break;
              case 'Profile': iconName = 'person-outline'; break;
            }
            return <Icon name={iconName} size={size} color={color} />;
          },
        })}
      >
        <Tab.Screen name="Home" component={HomeScreen} />
        <Tab.Screen name="Popular" component={PopularScreen} />
        <Tab.Screen
          name="Scan"
          component={ScanScreen}
          options={{
            tabBarIcon: ({ color }) => (
              <View style={styles.scanButton}>
                <Icon name="scan-outline" size={26} color="#fff" />
              </View>
            ),
          }}
        />
        <Tab.Screen name="Collection" component={CollectionScreen} />
        <Tab.Screen name="Profile" component={ProfileScreen} />
      </Tab.Navigator>
    </NavigationContainer>
  );
}

// --- Styles ---
const styles = StyleSheet.create({
  screen: { flex: 1, backgroundColor: '#111', paddingTop: 16 },
  centered: { flex: 1, backgroundColor: '#111', justifyContent: 'center', alignItems: 'center' },
  searchContainer: {
    backgroundColor: '#222',
    marginHorizontal: 12,
    marginBottom: 12,
    borderRadius: 10,
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 10,
  },
  searchInput: {
    flex: 1,
    height: 44,
    color: '#fff',
  },
  cameraIcon: {
    padding: 6,
    backgroundColor: '#333',
    borderRadius: 6,
  },
  sectionTitle: {
    color: '#fff',
    fontSize: 18,
    fontWeight: 'bold',
    marginLeft: 14,
    marginBottom: 10,
  },
  card: {
    backgroundColor: '#222',
    borderRadius: 12,
    marginRight: 14,
    padding: 10,
    width: 150,
    alignItems: 'center',
  },
  cardImage: {
    width: 120,
    height: 160,
    resizeMode: 'contain',
    borderRadius: 8,
  },
  cardTitle: {
    color: '#fff',
    marginTop: 6,
    fontWeight: '600',
  },
  tabBar: {
    backgroundColor: '#000',
    borderTopWidth: 0,
    height: 60,
  },
  scanButton: {
    backgroundColor: '#5f2efc',
    borderRadius: 40,
    padding: 12,
    bottom: 10,
  },
});
