import "@cloudscape-design/global-styles/index.css"
import { ContentLayout, Header, SpaceBetween } from "@cloudscape-design/components";
import { AboutSection } from "./components/AboutSection";
import { RankingSettingsSection } from "./components/RankingSettingsSection";

function App() {
  return (
    <ContentLayout
      defaultPadding
      headerVariant="high-contrast"
      header={
        <Header
          variant="h1"
          description="By Dana Goldberg and Emanuel Goldman"
        >
          Stable Chess GM Rankings
        </Header>
      }
    >
      <SpaceBetween size="m">
        <AboutSection/>
        <RankingSettingsSection/>
      </SpaceBetween>
    </ContentLayout>
    
  );
}

export default App;
