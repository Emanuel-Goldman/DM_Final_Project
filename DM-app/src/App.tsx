import "@cloudscape-design/global-styles/index.css"
import { ContentLayout, Header, SpaceBetween } from "@cloudscape-design/components";
import { AboutSection } from "./components/AboutSection";
import { UserSelectionSection } from "./components/UserSelectionSection";
import { DatasetPreviewSection } from "./components/DatasetPreviewSection";
import { Results } from "./components/Results";

function App() {
  return (
    <ContentLayout
      defaultPadding
      headerVariant="high-contrast"
      header={
        <Header
          variant="h1"
          description="Interactive tool for computing and analyzing stable rankings of Grandmaster chess players based on the On Obtaining Stable Rankings framework."
        >
          Stable Chess GM Rankings
        </Header>
      }
    >
      <SpaceBetween size="m">
        <AboutSection/>
        <DatasetPreviewSection/>
        <UserSelectionSection/>
        <Results/>
      </SpaceBetween>
    </ContentLayout>
    
  );
}

// function App() {
//   const [response, setResponse] = useState<any>(null);
//   const [loading, setLoading] = useState(false);
//   const [error, setError] = useState<string | null>(null);

//   const fetchData = async () => {
//     setLoading(true);
//     setError(null);
//     try {
//       const data = {
//         constraints: [[1, 2, "<="], [1, 1, ">="]],
//         method: "Ray Sweeping",
//         columns: ["rapid_win", "bullet_win"],
//         num_ret_tuples: 6,
//         num_of_rankings: 1,
//         num_of_samples: 1000,
//       };

//       const res = await axios.post("http://127.0.0.1:8000/ranking", data);
//       setResponse(res.data);
//     } catch (err: any) {
//       setError("Error fetching data");
//       console.error("API Error:", err);
//     } finally {
//       setLoading(false);
//     }
//   };

//   return (
//     <div style={{ textAlign: "center", padding: "20px" }}>
//       <h1>Data Management Project</h1>
//       <button onClick={fetchData} disabled={loading}>
//         {loading ? "Loading..." : "Make API Call"}
//       </button>
//       {error && <p style={{ color: "red" }}>{error}</p>}
//       {response && <pre>{JSON.stringify(response, null, 2)}</pre>}
//     </div>
//   );
// }

export default App;
